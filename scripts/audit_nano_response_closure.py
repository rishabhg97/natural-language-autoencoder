#!/usr/bin/env python3
"""Audit where generated explanations close and select a response-token cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nano_response_closure_audit.v1"


class ResponseClosureAuditError(ValueError):
    """Raised when generated response closure cannot be audited safely."""


def _real_payload(record: dict[str, Any]) -> dict[str, Any]:
    value = (record.get("controls") or {}).get("real")
    if not isinstance(value, dict):
        raise ResponseClosureAuditError("generated row is missing controls.real")
    return value


def closure_token_position(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    close_text: str,
) -> int | None:
    real = _real_payload(record)
    parsed = real.get("parsed") or {}
    if not isinstance(parsed, dict) or not bool(parsed.get("closed")):
        return None
    generated = str(real.get("generated") or "")
    close_start = generated.find(close_text)
    if close_start < 0:
        return None
    prefix = generated[: close_start + len(close_text)]
    return len(tokenizer.encode(prefix, add_special_tokens=False))


def audit_response_closure(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    split_limits: dict[str, int],
    candidate_caps: tuple[int, ...] = (150, 192, 224, 256),
    required_fraction: float = 0.95,
    close_text: str = "</explanation>",
) -> dict[str, Any]:
    if not candidate_caps or tuple(sorted(set(candidate_caps))) != candidate_caps:
        raise ResponseClosureAuditError("candidate_caps must be unique and increasing")
    if not 0.0 < required_fraction <= 1.0:
        raise ResponseClosureAuditError("required_fraction must be in (0, 1]")

    split_reports: dict[str, Any] = {}
    for split, limit in split_limits.items():
        selected = sorted(
            (record for record in records if str(record.get("split")) == split),
            key=lambda record: int(record.get("row_index", -1)),
        )[: int(limit)]
        if len(selected) != int(limit):
            raise ResponseClosureAuditError(
                f"{split} has {len(selected)} rows; expected {limit}"
            )
        positions = [
            closure_token_position(record, tokenizer=tokenizer, close_text=close_text)
            for record in selected
        ]
        closed_positions = [position for position in positions if position is not None]
        split_reports[split] = {
            "row_count": len(selected),
            "closed_count": len(closed_positions),
            "closed_fraction": len(closed_positions) / len(selected),
            "closed_by_cap": {
                str(cap): sum(position is not None and position <= cap for position in positions)
                / len(selected)
                for cap in candidate_caps
            },
            "closure_token_position": positions,
        }

    selected_cap = candidate_caps[-1]
    meets_required = False
    for cap in candidate_caps:
        if all(
            report["closed_by_cap"][str(cap)] >= required_fraction
            for report in split_reports.values()
        ):
            selected_cap = cap
            meets_required = True
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "required_fraction": required_fraction,
        "candidate_caps": list(candidate_caps),
        "selected_cap": selected_cap,
        "selected_cap_meets_required_fraction": meets_required,
        "close_text": close_text,
        "splits": split_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--candidate-cap", type=int, action="append", dest="caps")
    parser.add_argument("--required-fraction", type=float, default=0.95)
    parser.add_argument("--close-text", default="</explanation>")
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    records = [
        json.loads(line)
        for line in args.generated_jsonl.read_text().splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    report = audit_response_closure(
        records,
        tokenizer=tokenizer,
        split_limits={
            "validation": args.validation_limit,
            "test": args.test_limit,
        },
        candidate_caps=tuple(args.caps or (150, 192, 224, 256)),
        required_fraction=args.required_fraction,
        close_text=args.close_text,
    )
    report["metadata"] = {
        "generated_jsonl": str(args.generated_jsonl),
        "tokenizer": args.tokenizer,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report_json": str(args.report_json),
        "selected_cap": report["selected_cap"],
        "selected_cap_meets_required_fraction": report[
            "selected_cap_meets_required_fraction"
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
