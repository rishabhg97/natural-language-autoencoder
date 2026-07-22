#!/usr/bin/env python3
"""Normalize Nano round-trip, invariance, functional, and review evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from nano_functional_eval_data import read_generated_jsonl


SCHEMA_VERSION = "nano_r33_validity_eval.v1"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
DEFAULT_INJECTION_MARKERS = (
    "NLA_ACTIVATION_MARKER",
    "<concept>",
    "</concept>",
)


class ValidityEvidenceError(ValueError):
    """Raised when validity evidence is absent, stale, or not row-aligned."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValidityEvidenceError(f"JSON report must be an object: {path}")
    return value


def _resolve(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _selected_generated(
    records: list[dict[str, Any]],
    *,
    split: str,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [record for record in records if str(record.get("split")) == split]
    selected.sort(key=lambda record: int(record.get("row_index", -1)))
    if len(selected) < limit:
        raise ValidityEvidenceError(
            f"generated records for {split} have {len(selected)} rows; require {limit}"
        )
    selected = selected[:limit]
    indices = [int(record.get("row_index", -1)) for record in selected]
    if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        raise ValidityEvidenceError(f"generated records for {split} have invalid row indices")
    return selected


def _row_keys(records: list[dict[str, Any]]) -> list[str]:
    return [str(int(record["row_index"])) for record in records]


def _content_family_ids(records: list[dict[str, Any]]) -> list[str]:
    values = [str(record.get("content_family_id") or "") for record in records]
    if any(not value for value in values):
        raise ValidityEvidenceError(
            "generated records require non-empty content_family_id values"
        )
    return values


def _roundtrip_split(
    report: dict[str, Any],
    *,
    split: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    split_report = (report.get("splits") or {}).get(split)
    if not isinstance(split_report, dict):
        raise ValidityEvidenceError(f"round-trip report is missing split={split}")
    row_indices = [str(int(value)) for value in split_report.get("row_indices", [])]
    if row_indices != expected_keys:
        raise ValidityEvidenceError(f"round-trip row identity mismatch for split={split}")
    losses = (split_report.get("rowwise_normalized_mse") or {}).get("av_real")
    if not isinstance(losses, list) or len(losses) != len(expected_keys):
        raise ValidityEvidenceError(f"round-trip av_real losses are incomplete for split={split}")
    return {"losses": [float(value) for value in losses], "report": split_report}


def _functional_split(
    report: dict[str, Any],
    *,
    split: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    variants = (((report.get("splits") or {}).get(split) or {}).get("variants") or {})
    output: dict[str, Any] = {}
    for variant in ("stored_gold", "sft", "candidate"):
        rows = ((variants.get(variant) or {}).get("rows") or [])
        if not isinstance(rows, list):
            raise ValidityEvidenceError(
                f"functional report {split}/{variant} rows must be a list"
            )
        ordered = sorted(rows, key=lambda row: int(row.get("row_index", -1)))
        keys = [str(int(row.get("row_index", -1))) for row in ordered]
        if keys != expected_keys:
            raise ValidityEvidenceError(
                f"functional row identity mismatch for split={split} variant={variant}"
            )
        output[variant] = [dict(row.get("metrics") or {}) for row in ordered]
    output["row_keys"] = expected_keys
    return output


def _invariance_split(
    report: dict[str, Any],
    *,
    split: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    raw_rows = [
        str(int(value))
        for value in (((report.get("raw") or {}).get("splits") or {}).get(split) or {}).get(
            "row_indices", []
        )
    ]
    if raw_rows != expected_keys:
        raise ValidityEvidenceError(f"invariance row identity mismatch for split={split}")
    transforms = report.get("transforms") or {}
    retention = {
        str(name): float((values.get(split) or {})["fve_retention"])
        for name, values in transforms.items()
    }
    if not retention:
        raise ValidityEvidenceError(f"invariance report has no transforms for split={split}")
    return {"row_keys": raw_rows, "retention": retention}


def _generated_text(record: dict[str, Any], control: str) -> str:
    value = (record.get("controls") or {}).get(control)
    if isinstance(value, dict):
        return str(value.get("generated") or value.get("explanation") or "")
    return str(record.get("generated") or record.get("explanation") or "")


def scan_generated_leakage(
    records: list[dict[str, Any]],
    *,
    control: str = "real",
    injection_markers: tuple[str, ...] = DEFAULT_INJECTION_MARKERS,
) -> dict[str, int]:
    injection_marker_count = 0
    cjk_count = 0
    for record in records:
        text = _generated_text(record, control)
        injection_marker_count += int(any(marker in text for marker in injection_markers))
        cjk_count += int(bool(CJK_RE.search(text)))
    return {
        "row_count": len(records),
        "injection_marker_count": injection_marker_count,
        "cjk_count": cjk_count,
    }


def _control_wins(
    roundtrip_split: dict[str, Any],
    *,
    controls: list[str],
) -> dict[str, float]:
    win_rates = roundtrip_split.get("rowwise_win_rates") or {}
    output: dict[str, float] = {}
    for control in controls:
        key = f"av_real_vs_av_{control}"
        value = win_rates.get(key)
        if not isinstance(value, dict) or value.get("candidate_better_fraction") is None:
            raise ValidityEvidenceError(f"round-trip report is missing control win rate {key}")
        output[control] = float(value["candidate_better_fraction"])
    return output


def _parse_health(roundtrip_split: dict[str, Any]) -> dict[str, float]:
    parse = (roundtrip_split.get("generation_parse") or {}).get("real")
    if not isinstance(parse, dict):
        raise ValidityEvidenceError("round-trip report is missing real generation parse health")
    return {
        "usable_fraction": float(parse["usable_fraction"]),
        "closed_fraction": float(parse["closed_fraction"]),
    }


def _qualitative_split(report: dict[str, Any], split: str) -> dict[str, int]:
    value = (report.get("splits") or {}).get(split)
    if not isinstance(value, dict):
        raise ValidityEvidenceError(f"qualitative report is missing split={split}")
    return {
        "row_count": int(value.get("row_count", 0)),
        "reviewed_count": int(value.get("reviewed_count", 0)),
        "flagged_count": int(value.get("flagged_count", -1)),
    }


def load_gate_bundle(config_path: str | Path, *, candidate_name: str) -> dict[str, Any]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValidityEvidenceError(f"config must use schema_version {SCHEMA_VERSION}")
    thresholds = config.get("thresholds")
    evaluations = config.get("evaluations")
    if not isinstance(thresholds, dict) or not isinstance(evaluations, dict):
        raise ValidityEvidenceError("config requires thresholds and evaluations mappings")
    sft_config = evaluations.get("sft")
    candidate_config = (evaluations.get("candidates") or {}).get(candidate_name)
    if not isinstance(sft_config, dict) or not isinstance(candidate_config, dict):
        raise ValidityEvidenceError(f"unknown or incomplete candidate: {candidate_name}")

    base = path.parent
    required_rows = int(thresholds["required_rows"])
    eval_splits = tuple(str(value) for value in config.get("eval_splits", ["validation"]))
    if not eval_splits or len(set(eval_splits)) != len(eval_splits) or not set(
        eval_splits
    ).issubset({"validation", "test"}):
        raise ValidityEvidenceError(
            "eval_splits must be a non-empty unique validation/test list"
        )
    controls = [str(value) for value in config.get("required_controls", [])]
    if not controls:
        raise ValidityEvidenceError("required_controls must not be empty")
    markers = tuple(
        str(value) for value in config.get("injection_markers", DEFAULT_INJECTION_MARKERS)
    )

    def evidence_path(section: dict[str, Any], key: str) -> Path:
        if not section.get(key):
            raise ValidityEvidenceError(f"evaluation section is missing {key}")
        resolved = _resolve(section[key], base=base)
        if not resolved.is_file():
            raise ValidityEvidenceError(f"evidence file does not exist: {resolved}")
        return resolved

    sft_generated_path = evidence_path(sft_config, "generated_jsonl")
    candidate_generated_path = evidence_path(candidate_config, "generated_jsonl")
    sft_roundtrip_path = evidence_path(sft_config, "roundtrip_report")
    candidate_roundtrip_path = evidence_path(candidate_config, "roundtrip_report")
    invariance_path = evidence_path(candidate_config, "invariance_report")
    functional_path = evidence_path(candidate_config, "functional_report")
    qualitative_path = evidence_path(candidate_config, "qualitative_report")

    sft_generated = read_generated_jsonl(sft_generated_path)
    candidate_generated = read_generated_jsonl(candidate_generated_path)
    sft_roundtrip = _read_json(sft_roundtrip_path)
    candidate_roundtrip = _read_json(candidate_roundtrip_path)
    invariance = _read_json(invariance_path)
    functional = _read_json(functional_path)
    qualitative = _read_json(qualitative_path)

    bundle_splits: dict[str, Any] = {}
    for split in eval_splits:
        selected_sft = _selected_generated(sft_generated, split=split, limit=required_rows)
        selected_candidate = _selected_generated(
            candidate_generated, split=split, limit=required_rows
        )
        sft_keys = _row_keys(selected_sft)
        candidate_keys = _row_keys(selected_candidate)
        sft_families = _content_family_ids(selected_sft)
        candidate_families = _content_family_ids(selected_candidate)
        if sft_families != candidate_families:
            raise ValidityEvidenceError(
                f"content-family identity mismatch for split={split}"
            )
        sft_roundtrip_split = _roundtrip_split(
            sft_roundtrip,
            split=split,
            expected_keys=sft_keys,
        )
        candidate_roundtrip_split = _roundtrip_split(
            candidate_roundtrip,
            split=split,
            expected_keys=candidate_keys,
        )
        functional_split = _functional_split(
            functional,
            split=split,
            expected_keys=candidate_keys,
        )
        invariance_split = _invariance_split(
            invariance,
            split=split,
            expected_keys=candidate_keys,
        )
        bundle_splits[split] = {
            "content_family_ids": candidate_families,
            "row_keys": {
                "sft": sft_keys,
                "candidate": candidate_keys,
                "invariance": invariance_split["row_keys"],
                "functional": functional_split["row_keys"],
            },
            "roundtrip": {
                "sft_nmse": sft_roundtrip_split["losses"],
                "candidate_nmse": candidate_roundtrip_split["losses"],
            },
            "functional": {
                "stored_gold": functional_split["stored_gold"],
                "sft": functional_split["sft"],
                "candidate": functional_split["candidate"],
            },
            "invariance_retention": invariance_split["retention"],
            "control_win_fractions": _control_wins(
                candidate_roundtrip_split["report"], controls=controls
            ),
            "parse": _parse_health(candidate_roundtrip_split["report"]),
            "leakage": scan_generated_leakage(
                selected_candidate,
                injection_markers=markers,
            ),
            "qualitative": _qualitative_split(qualitative, split),
        }

    return {
        "candidate_name": candidate_name,
        "eval_splits": list(eval_splits),
        "splits": bundle_splits,
        "thresholds": thresholds,
        "evidence": {
            "config": str(path),
            "sft_generated_jsonl": str(sft_generated_path),
            "candidate_generated_jsonl": str(candidate_generated_path),
            "sft_roundtrip_report": str(sft_roundtrip_path),
            "candidate_roundtrip_report": str(candidate_roundtrip_path),
            "invariance_report": str(invariance_path),
            "functional_report": str(functional_path),
            "qualitative_report": str(qualitative_path),
        },
    }
