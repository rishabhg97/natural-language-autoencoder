#!/usr/bin/env python3
"""Convert deterministic structural flags into explicit panel decisions.

This is an unattended syntax/degeneration gate. It is deliberately labeled as
non-semantic and does not replace later human inspection of hero outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class StructuralReviewError(ValueError):
    """Raised when an unreviewed panel cannot be converted safely."""


def build_structural_reviews(panel: dict[str, Any]) -> dict[str, Any]:
    splits = panel.get("splits")
    if not isinstance(splits, dict):
        raise StructuralReviewError("panel requires a splits mapping")
    decisions: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("validation", "test"):
        rows = (splits.get(split) or {}).get("rows")
        if not isinstance(rows, list) or not rows:
            raise StructuralReviewError(f"panel split {split!r} has no rows")
        flagged_count = 0
        for row in rows:
            if not isinstance(row, dict) or "row_index" not in row:
                raise StructuralReviewError(f"invalid panel row in split {split!r}")
            reasons = row.get("automatic_flag_reasons")
            if not isinstance(reasons, list):
                raise StructuralReviewError(
                    f"panel row {split}:{row['row_index']} lacks automatic flags"
                )
            key = f"{split}:{int(row['row_index'])}"
            if key in decisions:
                raise StructuralReviewError(f"duplicate panel decision key: {key}")
            flagged = bool(reasons)
            flagged_count += int(flagged)
            decisions[key] = {
                "flagged": flagged,
                "notes": (
                    "automatic structural flags: " + ", ".join(str(reason) for reason in reasons)
                    if reasons
                    else "no automatic structural flags"
                ),
                "review_mode": "automatic_structural_v1",
            }
        split_counts[split] = {
            "rows": len(rows),
            "flagged": flagged_count,
        }
    return {
        "schema_version": "nano_qualitative_review_decisions.v1",
        "review_mode": "automatic_structural_v1",
        "limitations": (
            "This is not a semantic human review; it gates only empty, encoded-looking, "
            "repetitive, and severe length-regression outputs detected by the panel builder."
        ),
        "split_counts": split_counts,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = args.panel_json.read_bytes()
    panel = json.loads(payload)
    reviews = build_structural_reviews(panel)
    reviews["source_panel"] = {
        "path": str(args.panel_json),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reviews, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(args.output_json),
        "review_mode": reviews["review_mode"],
        "split_counts": reviews["split_counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
