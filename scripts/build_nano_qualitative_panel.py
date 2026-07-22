#!/usr/bin/env python3
"""Build a deterministic, reviewable R33 generated-text comparison panel."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nano_r33_qualitative_panel.v1"
ENCODED_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9+/])")
DOC_SUFFIX_RE = re.compile(r"(?:[:/_-]?\d+)+$")


class QualitativePanelError(ValueError):
    """Raised when a qualitative panel cannot be aligned or completed."""


def _real_text(record: dict[str, Any]) -> str:
    real = (record.get("controls") or {}).get("real") or {}
    parsed = real.get("parsed") if isinstance(real, dict) else None
    if isinstance(parsed, dict) and parsed.get("explanation") is not None:
        return str(parsed["explanation"]).strip()
    if isinstance(real, dict):
        return str(real.get("generated") or "").strip()
    return ""


def _document_type(doc_id: Any) -> str:
    value = str(doc_id or "unknown")
    stripped = DOC_SUFFIX_RE.sub("", value).rstrip(":/_-")
    return stripped or value


def _rank_bins(rows: list[dict[str, Any]], key: str, *, bin_count: int = 4) -> dict[int, int]:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (float(item[1][key]), int(item[1]["row_index"])),
    )
    size = len(ordered)
    return {
        original_index: min(bin_count - 1, rank * bin_count // max(size, 1))
        for rank, (original_index, _) in enumerate(ordered)
    }


def select_stratified_panel(
    rows: list[dict[str, Any]],
    *,
    panel_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select rows round-robin across document type and numeric quantile strata."""
    if panel_size <= 0:
        raise QualitativePanelError("panel_size must be positive")
    if len(rows) < panel_size:
        raise QualitativePanelError(
            f"only {len(rows)} aligned rows are available for panel_size={panel_size}"
        )
    bins = {
        key: _rank_bins(rows, key)
        for key in ("token_position", "activation_norm", "explanation_length")
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        stratum = (
            str(row["doc_type"]),
            bins["token_position"][index],
            bins["activation_norm"][index],
            bins["explanation_length"][index],
        )
        enriched = dict(row)
        enriched["stratum"] = {
            "doc_type": stratum[0],
            "token_position_bin": stratum[1],
            "activation_norm_bin": stratum[2],
            "explanation_length_bin": stratum[3],
        }
        groups[stratum].append(enriched)

    rng = random.Random(seed)
    by_doc: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for stratum, values in groups.items():
        values.sort(key=lambda row: int(row["row_index"]))
        rng.shuffle(values)
        by_doc[str(stratum[0])].append(stratum)
    for strata in by_doc.values():
        strata.sort(key=repr)
        rng.shuffle(strata)

    doc_types = sorted(by_doc)
    cursors = {doc_type: 0 for doc_type in doc_types}
    selected: list[dict[str, Any]] = []
    while len(selected) < panel_size:
        made_progress = False
        for doc_type in doc_types:
            strata = by_doc[doc_type]
            for offset in range(len(strata)):
                position = (cursors[doc_type] + offset) % len(strata)
                stratum = strata[position]
                if groups[stratum]:
                    selected.append(groups[stratum].pop())
                    cursors[doc_type] = (position + 1) % len(strata)
                    made_progress = True
                    break
            if len(selected) == panel_size:
                break
        if not made_progress:
            raise QualitativePanelError("stratified selection exhausted unexpectedly")
    return sorted(selected, key=lambda row: int(row["row_index"]))


def automatic_flag_reasons(candidate_text: str, sft_text: str) -> list[str]:
    """Return conservative machine-review hints; these never replace manual review."""
    reasons: list[str] = []
    candidate = candidate_text.strip()
    reference = sft_text.strip()
    if not candidate:
        reasons.append("empty")
    if ENCODED_RE.search(candidate):
        reasons.append("encoded_looking")

    words = re.findall(r"\w+", candidate.lower())
    units = [
        " ".join(unit.lower().split())
        for unit in re.split(r"(?<=[.!?])\s+|\n+", candidate)
        if unit.strip()
    ]
    duplicate_units = len(units) != len(set(units))
    repeated_coverage = 0.0
    if len(words) >= 12:
        fourgrams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
        if fourgrams:
            most_common = max(fourgrams.count(value) for value in set(fourgrams))
            repeated_coverage = most_common * 4 / len(words)
    if duplicate_units or repeated_coverage >= 0.25:
        reasons.append("repetition")

    reference_words = re.findall(r"\w+", reference)
    if len(reference_words) >= 10:
        ratio = len(words) / len(reference_words)
        if ratio < 0.2 or ratio > 5.0:
            reasons.append("length_regression")
    return reasons


def _activation_norm(source_row: dict[str, Any]) -> float:
    vector = source_row.get("activation_vector")
    if not isinstance(vector, (list, tuple)) or not vector:
        raise QualitativePanelError("source row is missing activation_vector")
    value = math.sqrt(sum(float(component) ** 2 for component in vector))
    if not math.isfinite(value):
        raise QualitativePanelError("source activation norm is non-finite")
    return value


def _source_text(source_row: dict[str, Any]) -> str:
    text = str(source_row.get("detokenized_text_truncated") or "").strip()
    if not text:
        raise QualitativePanelError("source row is missing detokenized source text")
    return text


def build_panel_report(
    candidate_records: list[dict[str, Any]],
    sft_records: list[dict[str, Any]] | None,
    *,
    source_rows_by_index: dict[int, dict[str, Any]],
    panel_size: int,
    seed: int,
    reference_mode: str = "sft_generated",
    review_decisions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if reference_mode not in {"sft_generated", "target_explanation"}:
        raise QualitativePanelError(f"unsupported reference_mode={reference_mode!r}")
    if reference_mode == "sft_generated" and not sft_records:
        raise QualitativePanelError(
            "sft_records are required for reference_mode=sft_generated"
        )
    review_decisions = review_decisions or {}
    sft_by_key = {
        (str(record.get("split")), int(record.get("row_index", -1))): record
        for record in (sft_records or [])
    }
    split_reports: dict[str, Any] = {}
    for split in ("validation", "test"):
        aligned: list[dict[str, Any]] = []
        seen: set[int] = set()
        for candidate in candidate_records:
            if str(candidate.get("split")) != split:
                continue
            row_index = int(candidate.get("row_index", -1))
            if row_index < 0 or row_index in seen:
                raise QualitativePanelError(f"invalid or duplicate {split} row_index={row_index}")
            seen.add(row_index)
            sft = sft_by_key.get((split, row_index))
            source = source_rows_by_index.get(row_index)
            if source is None:
                raise QualitativePanelError(
                    f"missing aligned source row for {split}:{row_index}"
                )
            candidate_text = _real_text(candidate)
            if reference_mode == "sft_generated":
                if sft is None:
                    raise QualitativePanelError(
                        f"missing aligned SFT row for {split}:{row_index}"
                    )
                reference_text = _real_text(sft)
            else:
                reference_text = str(candidate.get("target_explanation") or "").strip()
                if not reference_text:
                    raise QualitativePanelError(
                        f"missing target explanation for {split}:{row_index}"
                    )
            aligned.append(
                {
                    "split": split,
                    "row_index": row_index,
                    "doc_id": str(candidate.get("doc_id") or source.get("doc_id") or ""),
                    "doc_type": _document_type(candidate.get("doc_id") or source.get("doc_id")),
                    "token_position": int(
                        candidate.get("token_position", source.get("token_position", -1))
                    ),
                    "activation_norm": _activation_norm(source),
                    "explanation_length": len(re.findall(r"\w+", candidate_text)),
                    "source_text": _source_text(source),
                    "candidate_text": candidate_text,
                    "reference_text": reference_text,
                    "reference_mode": reference_mode,
                    "sft_text": reference_text,
                    "automatic_flag_reasons": automatic_flag_reasons(
                        candidate_text, reference_text
                    ),
                }
            )
        selected = select_stratified_panel(aligned, panel_size=panel_size, seed=seed)
        reviewed_count = 0
        flagged_count = 0
        for row in selected:
            decision = review_decisions.get(f"{split}:{row['row_index']}")
            if isinstance(decision, dict) and isinstance(decision.get("flagged"), bool):
                row["review"] = {
                    "status": "reviewed",
                    "flagged": bool(decision["flagged"]),
                    "notes": str(decision.get("notes") or ""),
                }
                reviewed_count += 1
                flagged_count += int(bool(decision["flagged"]))
            else:
                row["review"] = {"status": "pending", "flagged": None, "notes": ""}
        split_reports[split] = {
            "row_count": len(selected),
            "reviewed_count": reviewed_count,
            "flagged_count": flagged_count if reviewed_count == len(selected) else -1,
            "automatic_flagged_count": sum(
                bool(row["automatic_flag_reasons"]) for row in selected
            ),
            "rows": selected,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_mode": reference_mode,
        "panel_size_per_split": panel_size,
        "seed": seed,
        "review_complete": all(
            split["reviewed_count"] == split["row_count"]
            for split in split_reports.values()
        ),
        "splits": split_reports,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise QualitativePanelError(f"generated JSONL is empty or invalid: {path}")
    return records


def _load_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, dict):
        raise QualitativePanelError("reviews JSON must contain a decisions mapping")
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-generated-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Repeat once per frozen generated split file.",
    )
    parser.add_argument(
        "--sft-generated-jsonl",
        type=Path,
        action="append",
        help="Repeat once per comparison generated split file.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("sft_generated", "target_explanation"),
        default="sft_generated",
    )
    parser.add_argument("--source-base-parquet", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reviews-json", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    from nano_r33_source_rows import provenance_key, resolve_source_rows

    candidate = [
        record
        for path in args.candidate_generated_jsonl
        for record in _read_jsonl(path)
    ]
    if args.reference_mode == "sft_generated" and args.sft_generated_jsonl is None:
        parser.error("--sft-generated-jsonl is required for sft_generated references")
    sft = (
        [
            record
            for path in args.sft_generated_jsonl
            for record in _read_jsonl(path)
        ]
        if args.sft_generated_jsonl is not None
        else None
    )
    resolved = resolve_source_rows(args.source_base_parquet, candidate)
    source_by_index = {
        int(record["row_index"]): resolved[provenance_key(record)]
        for record in candidate
    }
    report = build_panel_report(
        candidate,
        sft,
        source_rows_by_index=source_by_index,
        panel_size=args.panel_size,
        seed=args.seed,
        reference_mode=args.reference_mode,
        review_decisions=_load_reviews(args.reviews_json),
    )
    report["metadata"] = {
        "candidate_generated_jsonl": [
            str(path) for path in args.candidate_generated_jsonl
        ],
        "sft_generated_jsonl": (
            [str(path) for path in args.sft_generated_jsonl]
            if args.sft_generated_jsonl
            else None
        ),
        "reference_mode": args.reference_mode,
        "source_base_parquet": str(args.source_base_parquet),
        "reviews_json": str(args.reviews_json) if args.reviews_json else None,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report_json": str(args.report_json),
        "review_complete": report["review_complete"],
        "splits": {
            name: {
                "row_count": value["row_count"],
                "reviewed_count": value["reviewed_count"],
                "automatic_flagged_count": value["automatic_flagged_count"],
            }
            for name, value in report["splits"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
