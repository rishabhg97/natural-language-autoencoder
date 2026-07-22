#!/usr/bin/env python3
"""Audit frozen NLA release text for privacy, secret, and copying risks."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import ipaddress
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = "nano_release_text_audit.v1"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=:-]{12,}",
            re.I,
        ),
    ),
)

INTERNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("workspace_path", re.compile(r"(?:^|\s)/(?:workspace|Users)/[^\s]+")),
    ("internal_s3", re.compile(r"\bs3://team-ipp-[^\s]+", re.I)),
    ("internal_hostname", re.compile(r"\b[A-Za-z0-9.-]+\.nvidia\.com\b", re.I)),
)


class ReleaseTextAuditError(ValueError):
    """Raised when an audit input violates the frozen protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReleaseTextAuditError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            yield value


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ReleaseTextAuditError(f"missing required field {dotted_path!r}")
        current = current[key]
    return current


def _luhn_valid(digits: str) -> bool:
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def scan_sensitive_text(text: str) -> dict[str, int]:
    """Return finding counts without retaining or returning matched values."""

    counts: Counter[str] = Counter()
    counts["email"] += len(EMAIL_RE.findall(text))
    counts["ssn"] += len(SSN_RE.findall(text))

    for match in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 10 <= len(digits) <= 15:
            counts["phone"] += 1

    for match in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _luhn_valid(digits):
            counts["payment_card"] += 1

    for match in IPV4_RE.finditer(text):
        try:
            address = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if address.is_private or address.is_loopback or address.is_link_local:
            counts["private_ipv4"] += 1

    for name, pattern in SECRET_PATTERNS + INTERNAL_PATTERNS:
        counts[name] += len(pattern.findall(text))
    return {name: count for name, count in sorted(counts.items()) if count}


def _words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def copy_statistics(candidate: str, source: str, *, min_block_words: int) -> dict[str, Any]:
    candidate_words = _words(candidate)
    source_words = _words(source)
    if not candidate_words or not source_words:
        return {
            "candidate_word_count": len(candidate_words),
            "source_word_count": len(source_words),
            "longest_contiguous_match_words": 0,
            "matched_block_words": 0,
            "matched_candidate_fraction": 0.0,
        }
    matcher = difflib.SequenceMatcher(None, candidate_words, source_words, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= min_block_words]
    longest = max((block.size for block in blocks), default=0)
    matched = sum(block.size for block in blocks)
    return {
        "candidate_word_count": len(candidate_words),
        "source_word_count": len(source_words),
        "longest_contiguous_match_words": longest,
        "matched_block_words": matched,
        "matched_candidate_fraction": matched / len(candidate_words),
    }


def _finding_metadata(record: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": record.get("split"),
        "row_index": record.get("row_index"),
        "doc_id": record.get("doc_id"),
        "content_family_id": record.get("content_family_id"),
    }


def _append_sensitive_findings(
    findings: list[dict[str, Any]],
    counts: Counter[str],
    *,
    record: Mapping[str, Any],
    dataset: str,
    field: str,
    text: str,
) -> None:
    for kind, count in scan_sensitive_text(text).items():
        counts[kind] += count
        findings.append(
            {
                **_finding_metadata(record, dataset),
                "field": field,
                "kind": kind,
                "count": count,
            }
        )


def run_audit(config: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseTextAuditError(f"schema_version must be {SCHEMA_VERSION!r}")
    paths = config.get("paths") or {}
    protocol = config.get("protocol") or {}
    generated_specs = paths.get("generated_jsonl") or []
    if not isinstance(generated_specs, list) or not generated_specs:
        raise ReleaseTextAuditError("paths.generated_jsonl must be a non-empty list")

    text_path = str(protocol.get("generated_text_path", "controls.real.parsed.explanation"))
    min_block_words = int(protocol.get("min_copy_block_words", 5))
    max_contiguous = int(protocol.get("max_contiguous_source_copy_words", 30))
    max_fraction = float(protocol.get("max_source_copy_fraction", 0.5))
    min_fraction_words = int(protocol.get("min_words_for_copy_fraction", 20))
    fail_kinds = set(protocol.get("fail_sensitive_kinds") or [])

    findings: list[dict[str, Any]] = []
    candidate_counts: Counter[str] = Counter()
    generated_rows = 0
    seen: set[tuple[Any, Any]] = set()
    input_hashes: dict[str, str] = {}

    for raw_spec in generated_specs:
        spec = {"path": raw_spec} if isinstance(raw_spec, str) else dict(raw_spec)
        path = Path(spec["path"])
        name = str(spec.get("name") or path.stem)
        input_hashes[name] = _sha256(path)
        for record in _read_jsonl(path):
            key = (record.get("split"), record.get("row_index"))
            if key in seen:
                raise ReleaseTextAuditError(f"duplicate generated row identity: {key!r}")
            seen.add(key)
            text = _nested(record, text_path)
            if not isinstance(text, str):
                raise ReleaseTextAuditError(f"{text_path!r} must resolve to text")
            generated_rows += 1
            _append_sensitive_findings(
                findings,
                candidate_counts,
                record=record,
                dataset=name,
                field="candidate_text",
                text=text,
            )

    panel_path = Path(paths["panel_json"])
    panel = json.loads(panel_path.read_text())
    input_hashes["panel_json"] = _sha256(panel_path)
    copy_findings: list[dict[str, Any]] = []
    source_reference_counts: Counter[str] = Counter()
    panel_rows = 0
    max_source_copy = 0
    max_reference_copy = 0
    max_source_fraction = 0.0
    max_reference_fraction = 0.0

    for split, split_payload in (panel.get("splits") or {}).items():
        for record in split_payload.get("rows") or []:
            panel_rows += 1
            candidate = str(record.get("candidate_text") or "")
            source = str(record.get("source_text") or "")
            reference = str(record.get("reference_text") or "")
            source_stats = copy_statistics(candidate, source, min_block_words=min_block_words)
            reference_stats = copy_statistics(candidate, reference, min_block_words=min_block_words)
            max_source_copy = max(max_source_copy, source_stats["longest_contiguous_match_words"])
            max_reference_copy = max(max_reference_copy, reference_stats["longest_contiguous_match_words"])
            max_source_fraction = max(max_source_fraction, source_stats["matched_candidate_fraction"])
            max_reference_fraction = max(max_reference_fraction, reference_stats["matched_candidate_fraction"])

            source_flag = source_stats["longest_contiguous_match_words"] > max_contiguous or (
                source_stats["candidate_word_count"] >= min_fraction_words
                and source_stats["matched_candidate_fraction"] > max_fraction
            )
            reference_flag = reference_stats["longest_contiguous_match_words"] > max_contiguous or (
                reference_stats["candidate_word_count"] >= min_fraction_words
                and reference_stats["matched_candidate_fraction"] > max_fraction
            )
            if source_flag or reference_flag:
                copy_findings.append(
                    {
                        **_finding_metadata({**record, "split": split}, "qualitative_panel"),
                        "source_copy_flag": source_flag,
                        "reference_copy_flag": reference_flag,
                        "source": source_stats,
                        "reference": reference_stats,
                    }
                )
            for field, text in (("source_text", source), ("reference_text", reference)):
                for kind, count in scan_sensitive_text(text).items():
                    source_reference_counts[f"{field}:{kind}"] += count

    failed_sensitive_counts = {
        kind: count for kind, count in candidate_counts.items() if kind in fail_kinds and count
    }
    fail_on_source_copy = bool(protocol.get("fail_on_source_copy", True))
    fail_on_reference_copy = bool(protocol.get("fail_on_reference_copy", False))
    failing_copy_rows = sum(
        bool(item["source_copy_flag"] and fail_on_source_copy)
        or bool(item["reference_copy_flag"] and fail_on_reference_copy)
        for item in copy_findings
    )
    automatic_gate_passed = not failed_sensitive_counts and failing_copy_rows == 0

    return {
        "schema_version": SCHEMA_VERSION,
        "automatic_gate_passed": automatic_gate_passed,
        "human_review_required": True,
        "claim_boundary": (
            "Automatic pattern and copying triage only. This report cannot establish "
            "privacy, factuality, consent, licensing, or absence of memorization."
        ),
        "config": str(config_path) if config_path else None,
        "input_sha256": input_hashes,
        "protocol": {
            "generated_text_path": text_path,
            "min_copy_block_words": min_block_words,
            "max_contiguous_source_copy_words": max_contiguous,
            "max_source_copy_fraction": max_fraction,
            "min_words_for_copy_fraction": min_fraction_words,
            "fail_sensitive_kinds": sorted(fail_kinds),
            "fail_on_source_copy": fail_on_source_copy,
            "fail_on_reference_copy": fail_on_reference_copy,
        },
        "generated_text": {
            "row_count": generated_rows,
            "finding_counts": dict(sorted(candidate_counts.items())),
            "failed_finding_counts": dict(sorted(failed_sensitive_counts.items())),
            "finding_rows": findings,
        },
        "qualitative_panel": {
            "row_count": panel_rows,
            "copy_flagged_rows": len(copy_findings),
            "failing_copy_rows": failing_copy_rows,
            "max_source_contiguous_match_words": max_source_copy,
            "max_reference_contiguous_match_words": max_reference_copy,
            "max_source_matched_candidate_fraction": max_source_fraction,
            "max_reference_matched_candidate_fraction": max_reference_fraction,
            "copy_findings": copy_findings,
            "source_reference_finding_counts": dict(sorted(source_reference_counts.items())),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    report = run_audit(config, config_path=args.config)
    output = Path(config["paths"]["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "automatic_gate_passed": report["automatic_gate_passed"],
                "generated_rows": report["generated_text"]["row_count"],
                "candidate_findings": report["generated_text"]["finding_counts"],
                "copy_flagged_rows": report["qualitative_panel"]["copy_flagged_rows"],
                "output_json": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
