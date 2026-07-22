#!/usr/bin/env python3
"""Build the frozen validation corpus and intervention grid for the NLA Observatory."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    git_revision,
    load_config,
    resolve_path,
    sha256_file,
    sha256_json,
    stable_int,
    write_json,
    write_jsonl,
)


ROW_SCHEMA = "nano_viz_row.v1"
INTERVENTION_SCHEMA = "nano_viz_intervention.v1"
SELECTION_SCHEMA = "nano_viz_selection.v1"
GRID_SCHEMA = "nano_viz_grid.v1"
REPORT_SCHEMA = "nano_viz_corpus_build_report.v1"
SECTION_LABELS = (
    "Syntax/continuation feature",
    "Discourse/semantic feature",
    "Genre/register feature",
    "Final-token constraint",
)
SECTION_PATTERN = re.compile(
    r"(?m)^(Syntax/continuation feature|Discourse/semantic feature|"
    r"Genre/register feature|Final-token constraint):\s*"
)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
CLAUSE_CHIPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("syntax", (0,)),
    ("discourse", (1,)),
    ("register", (2,)),
    ("final_token", (3,)),
    ("syntax_final", (0, 3)),
    ("discourse_register", (1, 2)),
)
PARAPHRASE_TYPES = (
    "compact_lines",
    "bullet_sections",
    "short_labels",
    "dash_delimiters",
    "rotated_sections",
    "normalized_spacing",
)


def parse_sections(text: str) -> tuple[str, ...]:
    matches = list(SECTION_PATTERN.finditer(text.strip()))
    labels = [match.group(1) for match in matches]
    if labels != list(SECTION_LABELS):
        raise ObservatoryConfigError(
            f"explanation must contain the four canonical sections in order; got {labels}"
        )
    values: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip()
        if not value:
            raise ObservatoryConfigError(f"section {match.group(1)!r} is empty")
        values.append(value)
    return tuple(values)


def render_sections(
    sections: tuple[str, ...] | list[str],
    *,
    labels: tuple[str, ...] | list[str] = SECTION_LABELS,
    delimiter: str = ": ",
    separator: str = "\n\n",
    prefix: str = "",
) -> str:
    return separator.join(
        f"{prefix}{label}{delimiter}{value.strip()}"
        for label, value in zip(labels, sections, strict=True)
    )


def _surface_paraphrase_body(value: str) -> str:
    replacements = (
        (r"\bExpects\b", "Anticipates"),
        (r"\bExpect\b", "Anticipate"),
        (r"\bSignals\b", "Indicates"),
        (r"\bReflects\b", "Uses"),
        (r"\brequires\b", "calls for"),
        (r"\brequiring\b", "calling for"),
        (r"\blikely\b", "probably"),
    )
    output = value
    for pattern, replacement in replacements:
        output = re.sub(pattern, replacement, output)
    if output == value:
        output = value.rstrip(".") + "."
    return output


def paraphrase_text(sections: tuple[str, ...], kind: str) -> str:
    if kind == "compact_lines":
        return render_sections(sections, separator="\n")
    if kind == "bullet_sections":
        return render_sections(sections, separator="\n", prefix="- ")
    if kind == "short_labels":
        return render_sections(
            sections,
            labels=("Syntax", "Meaning", "Register", "Next-token constraint"),
        )
    if kind == "dash_delimiters":
        return render_sections(sections, delimiter=" - ")
    if kind == "rotated_sections":
        order = (1, 2, 3, 0)
        return render_sections(
            tuple(sections[index] for index in order),
            labels=tuple(SECTION_LABELS[index] for index in order),
        )
    if kind == "normalized_spacing":
        normalized = tuple(re.sub(r"\s+", " ", value).strip() for value in sections)
        return render_sections(normalized)
    raise ObservatoryConfigError(f"unknown paraphrase type: {kind}")


def _replace_word(text: str, word_index: int) -> str:
    matches = list(WORD_PATTERN.finditer(text))
    match = matches[word_index]
    return (text[: match.start()] + "[MASK]" + text[match.end() :]).strip()


def _sample_word_indices(text: str, limit: int) -> list[int]:
    word_count = len(list(WORD_PATTERN.finditer(text)))
    if word_count <= limit:
        return list(range(word_count))
    return sorted(
        {
            min(word_count - 1, int(round(position * (word_count - 1) / (limit - 1))))
            for position in range(limit)
        }
    )


def _truncate_words(text: str, fraction: float) -> str:
    matches = list(WORD_PATTERN.finditer(text))
    keep = max(1, min(len(matches), int(math.ceil(len(matches) * fraction))))
    return text[: matches[keep - 1].end()].strip()


def _corrupt_words(text: str, *, kind: str, rate: float, seed: int) -> str:
    words = text.split()
    count = max(1, min(len(words), int(round(len(words) * rate))))
    ranked = sorted(range(len(words)), key=lambda index: stable_int(seed, kind, rate, index))
    selected = sorted(ranked[:count])
    if kind == "delete":
        selected_set = set(selected)
        return " ".join(word for index, word in enumerate(words) if index not in selected_set)
    if kind == "shuffle":
        replacement = list(reversed([words[index] for index in selected]))
        output = list(words)
        for index, value in zip(selected, replacement, strict=True):
            output[index] = value
        return " ".join(output)
    raise ObservatoryConfigError(f"unknown corruption kind: {kind}")


def load_panel(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "nano_r33_qualitative_panel.v1":
        raise ObservatoryConfigError("qualitative panel has an unexpected schema")
    validation = (payload.get("splits") or {}).get("validation") or {}
    rows = validation.get("rows") or []
    if int(validation.get("row_count", -1)) != len(rows):
        raise ObservatoryConfigError("qualitative panel validation row_count mismatch")
    return rows


def load_family_lookup(path: Path) -> dict[int, dict[str, str]]:
    with np.load(path, allow_pickle=False) as cache:
        required = (
            "validation__row_indices",
            "validation__content_family_ids",
            "validation__doc_ids",
        )
        missing = [key for key in required if key not in cache]
        if missing:
            raise ObservatoryConfigError(f"prediction cache is missing {missing}")
        row_indices = cache[required[0]]
        family_ids = cache[required[1]]
        doc_ids = cache[required[2]]
        if not (len(row_indices) == len(family_ids) == len(doc_ids)):
            raise ObservatoryConfigError("prediction cache identity arrays disagree")
        return {
            int(row_index): {
                "content_family_id": str(family_id),
                "doc_id": str(doc_id),
            }
            for row_index, family_id, doc_id in zip(
                row_indices, family_ids, doc_ids, strict=True
            )
        }


def build_rows(
    panel_rows: list[dict[str, Any]],
    family_lookup: dict[int, dict[str, str]],
    generated_lookup: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for panel_row in panel_rows:
        row_index = int(panel_row["row_index"])
        if row_index in seen:
            raise ObservatoryConfigError(f"duplicate panel row_index {row_index}")
        seen.add(row_index)
        identity = family_lookup.get(row_index)
        if identity is None:
            raise ObservatoryConfigError(
                f"panel row {row_index} is absent from the qualified validation cache"
            )
        doc_id = str(panel_row["doc_id"])
        if identity["doc_id"] != doc_id:
            raise ObservatoryConfigError(
                f"doc_id mismatch for row {row_index}: {doc_id!r} != {identity['doc_id']!r}"
            )
        target = str(panel_row["reference_text"]).strip()
        candidate = str(panel_row["candidate_text"]).strip()
        parse_sections(target)
        parse_sections(candidate)
        token_position = int(panel_row["token_position"])
        generated = (generated_lookup or {}).get(row_index)
        if generated is not None and str(generated.get("doc_id")) != doc_id:
            raise ObservatoryConfigError(
                f"generated provenance doc_id mismatch for row {row_index}"
            )
        n_raw_tokens = (
            int(generated["n_raw_tokens"])
            if generated is not None and generated.get("n_raw_tokens") is not None
            else token_position + 1
        )
        output.append(
            {
                "schema_version": ROW_SCHEMA,
                "population": "QUALIFIED",
                "claim_scope": "stored_snapshot",
                "row_id": f"validation-{row_index}",
                "split": "validation",
                "row_index": row_index,
                "doc_id": doc_id,
                "token_position": token_position,
                "n_raw_tokens": n_raw_tokens,
                "source_row_index": (
                    int(generated["source_row_index"])
                    if generated is not None and generated.get("source_row_index") is not None
                    else None
                ),
                "content_family_id": identity["content_family_id"],
                "source_text": str(panel_row["source_text"]),
                "target_explanation": target,
                "av_explanation": candidate,
                "activation_norm": float(panel_row["activation_norm"]),
                "stratum": panel_row.get("stratum") or {},
                "source_text_release_status": "privacy_cleared_panel",
            }
        )
    output.sort(key=lambda row: int(row["row_index"]))
    return output


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    behavior_count: int,
    canary_count: int,
    film_count: int,
    film_min_position: int,
) -> dict[str, list[str]]:
    ranked = sorted(rows, key=lambda row: stable_int(seed, "behavior", row["row_id"]))
    behavior = ranked[:behavior_count]
    behavior_ids = {row["row_id"] for row in behavior}
    canary_behavior = sorted(
        behavior, key=lambda row: stable_int(seed, "canary_behavior", row["row_id"])
    )[: canary_count // 2]
    nonbehavior = [row for row in rows if row["row_id"] not in behavior_ids]
    canary_other = sorted(
        nonbehavior, key=lambda row: stable_int(seed, "canary_other", row["row_id"])
    )[: canary_count - len(canary_behavior)]
    canary = canary_behavior + canary_other

    eligible = [
        row
        for row in rows
        if int(row["token_position"]) >= film_min_position
        and len(WORD_PATTERN.findall(row["source_text"])) >= film_min_position
    ]
    eligible.sort(
        key=lambda row: (
            -len(WORD_PATTERN.findall(row["source_text"])),
            stable_int(seed, "film", row["row_id"]),
        )
    )
    film: list[dict[str, Any]] = []
    film_docs: set[str] = set()
    for row in eligible:
        if row["doc_id"] in film_docs:
            continue
        film.append(row)
        film_docs.add(row["doc_id"])
        if len(film) == film_count:
            break
    if len(film) != film_count:
        raise ObservatoryConfigError(
            f"film selection found {len(film)} eligible unique docs, need {film_count}"
        )
    return {
        "deep_dive_row_ids": [row["row_id"] for row in rows],
        "behavior_row_ids": sorted(behavior_ids),
        "canary_row_ids": sorted(row["row_id"] for row in canary),
        "film_row_ids": sorted(row["row_id"] for row in film),
    }


def _cell(
    *,
    row: dict[str, Any],
    family: str,
    variant: str,
    text: str | None,
    depth: str,
    spec: dict[str, Any],
    control_group_id: str | None = None,
    state: str = "ready",
) -> dict[str, Any]:
    identity = {
        "row_id": row["row_id"],
        "family": family,
        "variant": variant,
        "spec": spec,
    }
    return {
        "schema_version": INTERVENTION_SCHEMA,
        "cell_id": f"cell-{sha256_json(identity)[:20]}",
        "row_id": row["row_id"],
        "row_index": row["row_index"],
        "family": family,
        "variant": variant,
        "depth": depth,
        "control_group_id": control_group_id,
        "state": state,
        "text": text,
        "text_sha256": sha256_json(text) if text is not None else None,
        "spec": spec,
    }


def build_interventions(
    rows: list[dict[str, Any]],
    selections: dict[str, list[str]],
    *,
    seed: int,
    word_occlusion_limit: int,
    truncation_points: int,
    corruption_rates: list[float],
    alternate_tellings: int,
) -> list[dict[str, Any]]:
    behavior_ids = set(selections["behavior_row_ids"])
    interventions: list[dict[str, Any]] = []
    for row_position, row in enumerate(rows):
        sections = parse_sections(row["target_explanation"])
        depth = "BEHAVIOR" if row["row_id"] in behavior_ids else "METRIC"
        interventions.append(
            _cell(
                row=row,
                family="identity",
                variant="teacher",
                text=row["target_explanation"],
                depth=depth,
                spec={"dose": 0.0},
            )
        )
        donor_order = sorted(
            [candidate for candidate in rows if candidate["row_id"] != row["row_id"]],
            key=lambda candidate: stable_int(seed, row["row_id"], candidate["row_id"]),
        )
        semantic_donor = donor_order[0]
        random_donor = donor_order[-1]
        semantic_sections = parse_sections(semantic_donor["target_explanation"])
        random_sections = parse_sections(random_donor["target_explanation"])
        doses = (0.5, 1.0) if depth == "BEHAVIOR" else (1.0,)
        for chip_name, section_indices in CLAUSE_CHIPS:
            group_id = f"cg-{sha256_json([row['row_id'], chip_name])[:16]}"
            lane_sections: dict[str, tuple[str, ...]] = {}
            edited = list(sections)
            placebo = list(sections)
            random_edit = list(sections)
            for section_index in section_indices:
                edited[section_index] = semantic_sections[section_index]
                placebo[section_index] = _surface_paraphrase_body(sections[section_index])
                random_edit[section_index] = random_sections[section_index]
            lane_sections["edit"] = tuple(edited)
            lane_sections["paraphrase_placebo"] = tuple(placebo)
            lane_sections["random_edit"] = tuple(random_edit)
            for dose in doses:
                for lane, values in lane_sections.items():
                    interventions.append(
                        _cell(
                            row=row,
                            family="clause_swap",
                            variant=f"{chip_name}:{lane}:a{dose:g}",
                            text=render_sections(values),
                            depth=depth,
                            control_group_id=group_id,
                            spec={
                                "chip": chip_name,
                                "lane": lane,
                                "dose": dose,
                                "section_indices": list(section_indices),
                                "semantic_donor_row_id": semantic_donor["row_id"],
                                "random_donor_row_id": random_donor["row_id"],
                            },
                        )
                    )

        for mask in range(16):
            kept = [sections[index] if mask & (1 << index) else "[ABLATE]" for index in range(4)]
            interventions.append(
                _cell(
                    row=row,
                    family="section_ablation",
                    variant=f"mask_{mask:04b}",
                    text=render_sections(kept),
                    depth="METRIC",
                    spec={"mask": mask, "kept_sections": [index for index in range(4) if mask & (1 << index)]},
                )
            )

        for word_index in _sample_word_indices(row["target_explanation"], word_occlusion_limit):
            interventions.append(
                _cell(
                    row=row,
                    family="word_occlusion",
                    variant=f"word_{word_index:03d}",
                    text=_replace_word(row["target_explanation"], word_index),
                    depth="METRIC",
                    spec={"word_index": word_index, "replacement": "[MASK]"},
                )
            )

        for point in range(1, truncation_points + 1):
            fraction = point / truncation_points
            interventions.append(
                _cell(
                    row=row,
                    family="truncation",
                    variant=f"prefix_{point:02d}_of_{truncation_points}",
                    text=_truncate_words(row["target_explanation"], fraction),
                    depth="METRIC",
                    spec={"fraction": fraction},
                )
            )

        for kind in PARAPHRASE_TYPES:
            interventions.append(
                _cell(
                    row=row,
                    family="paraphrase",
                    variant=kind,
                    text=paraphrase_text(sections, kind),
                    depth="METRIC",
                    spec={"kind": kind, "semantic_intent": "surface_preserving"},
                )
            )

        for corruption_kind in ("shuffle", "delete"):
            for rate in corruption_rates:
                interventions.append(
                    _cell(
                        row=row,
                        family="corruption",
                        variant=f"{corruption_kind}_{rate:g}",
                        text=_corrupt_words(
                            row["target_explanation"],
                            kind=corruption_kind,
                            rate=rate,
                            seed=stable_int(seed, row["row_id"], corruption_kind, rate),
                        ),
                        depth="METRIC",
                        spec={"kind": corruption_kind, "rate": rate},
                    )
                )

        for telling_index in range(alternate_tellings):
            interventions.append(
                _cell(
                    row=row,
                    family="alternate_telling",
                    variant=f"sample_{telling_index:02d}",
                    text=None,
                    depth="METRIC",
                    state="pending_model_generation",
                    spec={
                        "sample_index": telling_index,
                        "seed": stable_int(seed, "telling", row_position, telling_index),
                    },
                )
            )
    return interventions


def validate_grid(
    rows: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    selections: dict[str, list[str]],
) -> dict[str, Any]:
    row_ids = {row["row_id"] for row in rows}
    if len(row_ids) != len(rows):
        raise ObservatoryConfigError("row IDs are not unique")
    cell_ids = [cell["cell_id"] for cell in interventions]
    if len(set(cell_ids)) != len(cell_ids):
        raise ObservatoryConfigError("cell IDs are not unique")
    if any(cell["row_id"] not in row_ids for cell in interventions):
        raise ObservatoryConfigError("an intervention references an unknown row")
    clause_groups: dict[str, set[str]] = {}
    for cell in interventions:
        if cell["family"] != "clause_swap" or float(cell["spec"]["dose"]) != 1.0:
            continue
        clause_groups.setdefault(cell["control_group_id"], set()).add(cell["spec"]["lane"])
    expected_lanes = {"edit", "paraphrase_placebo", "random_edit"}
    incomplete = {group: lanes for group, lanes in clause_groups.items() if lanes != expected_lanes}
    if incomplete:
        raise ObservatoryConfigError(f"incomplete clause control groups: {incomplete}")
    for key in ("deep_dive_row_ids", "behavior_row_ids", "canary_row_ids", "film_row_ids"):
        if not set(selections[key]).issubset(row_ids):
            raise ObservatoryConfigError(f"selection {key} contains unknown rows")
    family_counts: dict[str, int] = {}
    for cell in interventions:
        family_counts[cell["family"]] = family_counts.get(cell["family"], 0) + 1
    return {
        "row_count": len(rows),
        "cell_count": len(interventions),
        "family_counts": dict(sorted(family_counts.items())),
        "control_group_count": len(clause_groups),
        "control_groups_complete": True,
    }


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    paths = config["paths"]
    selection_cfg = config["selection"]
    grid_cfg = config["grid"]
    panel_path = resolve_path(paths["qualitative_panel_json"], config_path=config_path)
    cache_path = resolve_path(paths["validation_prediction_cache_npz"], config_path=config_path)
    output_dir = output_override or resolve_path(paths["corpus_dir"], config_path=config_path)

    generated_lookup = None
    generated_path = None
    generated_path_value = paths.get("generated_validation_jsonl")
    if generated_path_value:
        generated_path = resolve_path(generated_path_value, config_path=config_path)
        if not generated_path.is_file():
            raise ObservatoryConfigError(
                f"configured generated validation JSONL is missing: {generated_path}"
            )
        generated_records = []
        with generated_path.open() as handle:
            for line in handle:
                if line.strip():
                    generated_records.append(json.loads(line))
        generated_lookup = {
            int(record["row_index"]): record
            for record in generated_records
            if str(record.get("split")) == "validation"
        }
    rows = build_rows(
        load_panel(panel_path),
        load_family_lookup(cache_path),
        generated_lookup,
    )
    if len(rows) != int(selection_cfg["deep_dive_rows"]):
        raise ObservatoryConfigError(
            f"deep-dive corpus must contain exactly {selection_cfg['deep_dive_rows']} rows"
        )
    selections = select_rows(
        rows,
        seed=int(selection_cfg["seed"]),
        behavior_count=int(selection_cfg["behavior_rows"]),
        canary_count=int(selection_cfg["canary_rows"]),
        film_count=int(selection_cfg["film_rows"]),
        film_min_position=int(selection_cfg.get("film_min_position", 130)),
    )
    interventions = build_interventions(
        rows,
        selections,
        seed=int(selection_cfg["seed"]),
        word_occlusion_limit=int(grid_cfg.get("word_occlusion_limit", 80)),
        truncation_points=int(grid_cfg.get("truncation_points", 10)),
        corruption_rates=[float(value) for value in grid_cfg.get("corruption_rates", [0.1, 0.25, 0.5])],
        alternate_tellings=int(grid_cfg.get("alternate_tellings", 8)),
    )
    grid_summary = validate_grid(rows, interventions, selections)
    config_sha = config_fingerprint(config)
    source_provenance = {
        "qualitative_panel_json": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
        },
        "validation_prediction_cache_npz": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
        },
    }
    if generated_path is not None:
        source_provenance["generated_validation_jsonl"] = {
            "path": str(generated_path),
            "sha256": sha256_file(generated_path),
        }
    selection_manifest = {
        "schema_version": SELECTION_SCHEMA,
        "config_sha256": config_sha,
        "seed": int(selection_cfg["seed"]),
        "population": "QUALIFIED",
        "claim_scope": "stored_snapshot",
        **selections,
        "source_provenance": source_provenance,
    }
    grid_spec = {
        "schema_version": GRID_SCHEMA,
        "config_sha256": config_sha,
        "grid": grid_cfg,
        "selection_sha256": sha256_json(selection_manifest),
        "summary": grid_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = write_jsonl(output_dir / "rows.jsonl", rows)
    cell_count = write_jsonl(output_dir / "interventions.jsonl", interventions)
    write_json(output_dir / "selection_manifest.json", selection_manifest)
    write_json(output_dir / "grid_spec.json", grid_spec)
    report = {
        "schema_version": REPORT_SCHEMA,
        "passed": row_count == 50 and cell_count == grid_summary["cell_count"],
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "code_revision": git_revision(config_path.resolve().parents[2]),
        "output_dir": str(output_dir),
        "rows": row_count,
        "interventions": cell_count,
        "grid_summary": grid_summary,
        "selection_manifest_sha256": sha256_json(selection_manifest),
        "grid_spec_sha256": sha256_json(grid_spec),
        "source_provenance": source_provenance,
    }
    write_json(output_dir / "corpus_build_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args.config, args.output_dir)
    except (OSError, ObservatoryConfigError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
