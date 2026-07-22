#!/usr/bin/env python3
"""Offline inference phases for the Observatory poetry-planning lens."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    read_jsonl,
    resolve_path,
    sha256_file,
    stable_int,
    write_json,
    write_jsonl,
)
from .model_runtime import (
    greedy_generate_patched_full_prefix,
    greedy_generate_unpatched,
    hf_checkpoint_complete,
    load_av_model,
    release_cuda_memory,
    rowwise_reconstruction_metrics,
    sample_generate_batch_full_prefix,
)


CONFIG_SCHEMA = "nano_viz_poetry_planning.v1"
REPORT_SCHEMA = "nano_viz_poetry_planning_report.v1"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservatoryConfigError(f"{name} must be a mapping")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ObservatoryConfigError(f"{name} must be a non-empty list")
    return value


def load_poetry_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    config = _mapping(config, name="config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ObservatoryConfigError(
            f"config schema_version must be {CONFIG_SCHEMA!r}"
        )
    for section in ("paths", "models", "evaluation", "gates"):
        _mapping(config.get(section), name=section)
    _list(config.get("cases"), name="cases")
    return config


def normalized_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def contains_term(text: str, terms: list[str]) -> bool:
    words = set(normalized_words(text))
    return any(str(term).lower() in words for term in terms)


def build_case_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate cases and materialize the exact causal prefixes."""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in _list(config.get("cases"), name="cases"):
        case = _mapping(raw, name="case")
        case_id = str(case.get("case_id") or "")
        if not CASE_ID_RE.fullmatch(case_id) or case_id in seen:
            raise ObservatoryConfigError(f"invalid or duplicate case_id: {case_id!r}")
        seen.add(case_id)
        framing = str(case.get("framing") or "A rhyming couplet:").strip()
        first_line = str(case.get("first_line") or "").strip()
        second_line = str(case.get("second_line") or "").strip()
        target_word = str(case.get("target_word") or "").strip().lower()
        target_terms = [
            str(value).strip().lower()
            for value in _list(case.get("target_terms"), name=f"{case_id}.target_terms")
        ]
        alternate_terms = [
            str(value).strip().lower()
            for value in _list(
                case.get("alternate_terms"), name=f"{case_id}.alternate_terms"
            )
        ]
        edit_map = {
            str(key).strip().lower(): str(value).strip()
            for key, value in _mapping(
                case.get("edit_map"), name=f"{case_id}.edit_map"
            ).items()
        }
        if not first_line or not second_line or not target_word:
            raise ObservatoryConfigError(f"{case_id} has empty required poetry text")
        if target_word not in target_terms:
            raise ObservatoryConfigError(
                f"{case_id}.target_terms must contain target_word {target_word!r}"
            )
        prefix_text = f"{framing}\n{first_line}\n"
        full_text = f"{prefix_text}{second_line}"
        prefix_words = set(normalized_words(prefix_text))
        leaked = sorted(prefix_words.intersection(target_terms))
        if leaked:
            raise ObservatoryConfigError(
                f"{case_id} leaks future target terms into its prefix: {leaked}"
            )
        if target_word not in normalized_words(second_line):
            raise ObservatoryConfigError(
                f"{case_id}.second_line must contain target_word {target_word!r}"
            )
        if not set(target_terms).intersection(edit_map):
            raise ObservatoryConfigError(
                f"{case_id}.edit_map must edit at least one target term"
            )
        if any(not key or not value for key, value in edit_map.items()):
            raise ObservatoryConfigError(f"{case_id}.edit_map contains an empty edit")
        output.append(
            {
                "case_id": case_id,
                "framing": framing,
                "first_line": first_line,
                "second_line": second_line,
                "prefix_text": prefix_text,
                "full_text": full_text,
                "cue": str(case.get("cue") or "").strip(),
                "target_word": target_word,
                "target_terms": target_terms,
                "alternate_terms": alternate_terms,
                "edit_map": edit_map,
            }
        )
    minimum = int(config["gates"].get("minimum_cases", 1))
    if len(output) < minimum:
        raise ObservatoryConfigError(
            f"poetry corpus has {len(output)} cases; minimum is {minimum}"
        )
    return output


def edit_explanation(text: str, edit_map: dict[str, str]) -> tuple[str, list[str]]:
    """Apply case-insensitive whole-word edits and report changed source terms."""

    edited = str(text)
    changed: list[str] = []
    for source, replacement in edit_map.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        edited, count = pattern.subn(replacement, edited)
        if count:
            changed.append(source)
    return edited, changed


def steering_replacement(
    gold: np.ndarray,
    delta: np.ndarray,
    dose: float,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Add a paper-style, gold-norm-scaled edit direction to an activation."""

    gold_array = np.asarray(gold, dtype=np.float32)
    delta_array = np.asarray(delta, dtype=np.float32)
    if gold_array.shape != delta_array.shape or gold_array.ndim != 1:
        raise ObservatoryConfigError("gold and delta must be matching vectors")
    if not np.isfinite(gold_array).all() or not np.isfinite(delta_array).all():
        raise ObservatoryConfigError("steering vectors must be finite")
    delta_norm = float(np.linalg.norm(delta_array))
    if delta_norm <= epsilon:
        raise ObservatoryConfigError("cannot steer with a zero-norm edit direction")
    gold_norm = float(np.linalg.norm(gold_array))
    return gold_array + float(dose) * gold_norm * delta_array / delta_norm


def _paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    output_root = resolve_path(config["paths"]["output_root"], config_path=config_path)
    return output_root, output_root / "reports"


def _report_reusable(
    path: Path, config: dict[str, Any], *, force: bool
) -> dict[str, Any] | None:
    if force or not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("passed") and report.get("config_sha256") == config_fingerprint(
        config
    ):
        return report
    return None


def _require_report(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ObservatoryConfigError(f"{label} report is missing: {path}")
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise ObservatoryConfigError(f"{label} did not pass")
    return report


def run_prepare(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    from transformers import PreTrainedTokenizerFast

    config = load_poetry_config(config_path)
    output_root, reports = _paths(config, config_path)
    report_path = reports / "poetry_prepare_report.json"
    if reusable := _report_reusable(report_path, config, force=force):
        return reusable
    cases = build_case_records(config)
    base_hf = resolve_path(config["models"]["base_hf"], config_path=config_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_hf, trust_remote_code=True, local_files_only=True
    )
    window = int(config["evaluation"]["positions_before_anchor"])
    if window < 1:
        raise ObservatoryConfigError("positions_before_anchor must be positive")
    prepared: list[dict[str, Any]] = []
    for case in cases:
        prefix_ids = [
            int(value)
            for value in tokenizer(
                case["prefix_text"], add_special_tokens=True
            )["input_ids"]
        ]
        full_ids = [
            int(value)
            for value in tokenizer(case["full_text"], add_special_tokens=True)[
                "input_ids"
            ]
        ]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ObservatoryConfigError(
                f"tokenization does not preserve the causal prefix for {case['case_id']}"
            )
        anchor = len(prefix_ids) - 1
        start = max(0, anchor - window)
        positions = list(range(start, anchor + 1))
        prepared.append(
            {
                **case,
                "prefix_token_ids": prefix_ids,
                "full_token_ids": full_ids,
                "anchor_position": anchor,
                "analysis_positions": positions,
                "analysis_token_text": [
                    tokenizer.decode([prefix_ids[position]]) for position in positions
                ],
            }
        )
    corpus_dir = output_root / "poetry_corpus"
    cases_path = corpus_dir / "cases.jsonl"
    write_jsonl(cases_path, prepared)
    report = {
        "schema_version": REPORT_SCHEMA,
        "phase": "poetry_prepare",
        "passed": len(prepared) == len(cases),
        "config_sha256": config_fingerprint(config),
        "cases": len(prepared),
        "positions": sum(len(case["analysis_positions"]) for case in prepared),
        "future_leakage_cases": 0,
        "cases_artifact": {"path": str(cases_path), "sha256": sha256_file(cases_path)},
    }
    write_json(report_path, report)
    return report


def _write_activation_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("case_id", pa.string()),
            ("position", pa.int64()),
            ("relative_offset", pa.int64()),
            ("token_id", pa.int64()),
            ("token_text", pa.string()),
            ("activation_vector", pa.list_(pa.float16(), 2688)),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
    )
    temporary.replace(path)


def run_extract(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import torch
    from transformers import PreTrainedTokenizerFast

    from nano_introspection import resolve_nano_module_paths
    from nano_r33_functional_runtime import load_target_model

    config = load_poetry_config(config_path)
    output_root, reports = _paths(config, config_path)
    report_path = reports / "poetry_extract_report.json"
    if reusable := _report_reusable(report_path, config, force=force):
        return reusable
    _require_report(reports / "poetry_prepare_report.json", label="poetry prepare")
    cases = read_jsonl(output_root / "poetry_corpus" / "cases.jsonl")
    models = config["models"]
    evaluation = config["evaluation"]
    base_hf = resolve_path(models["base_hf"], config_path=config_path)
    if not hf_checkpoint_complete(base_hf):
        raise ObservatoryConfigError(f"complete base checkpoint required: {base_hf}")
    args = SimpleNamespace(
        target_model=str(base_hf),
        target_torch_dtype=str(evaluation["torch_dtype"]),
        target_trust_remote_code=True,
        target_local_files_only=True,
        target_revision=None,
        target_device_map=str(evaluation.get("base_device_map", "auto")),
    )
    model = load_target_model(args)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_hf, trust_remote_code=True, local_files_only=True
    )
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    boundary = int(evaluation["boundary"])
    if layers is None or not 1 <= boundary <= len(layers):
        raise ObservatoryConfigError(f"invalid poetry boundary: {boundary}")
    boundary_module = layers[boundary - 1]
    start_device = model.get_input_embeddings().weight.device
    shard_dir = output_root / "poetry_extract" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in shard_dir.glob("*.json"):
            path.unlink()
    try:
        for ordinal, case in enumerate(cases, start=1):
            shard_path = shard_dir / f"{case['case_id']}.json"
            if shard_path.is_file():
                continue
            print(
                f"[observatory:poetry] extract {ordinal}/{len(cases)} "
                f"case={case['case_id']}",
                flush=True,
            )
            prefix = [int(value) for value in case["prefix_token_ids"]]
            input_ids = torch.tensor([prefix], dtype=torch.long, device=start_device)
            attention_mask = torch.ones_like(input_ids)
            captured: dict[str, Any] = {}

            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured["hidden"] = hidden.detach().float().cpu()

            handle = boundary_module.register_forward_hook(capture)
            try:
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()
            hidden = captured.get("hidden")
            if hidden is None or tuple(hidden.shape) != (1, len(prefix), 2688):
                raise ObservatoryConfigError(
                    f"unexpected poetry trace shape for {case['case_id']}: "
                    f"{None if hidden is None else tuple(hidden.shape)}"
                )
            continuation_ids = greedy_generate_unpatched(
                model=model,
                tokenizer=tokenizer,
                prefix=prefix,
                max_new_tokens=int(evaluation["continuation_tokens"]),
                pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
                eos_token_id=tokenizer.eos_token_id,
                backend=str(evaluation.get("generation_backend", "full_prefix")),
            )
            anchor = int(case["anchor_position"])
            rows = []
            for position in case["analysis_positions"]:
                position = int(position)
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "position": position,
                        "relative_offset": position - anchor,
                        "token_id": prefix[position],
                        "token_text": tokenizer.decode([prefix[position]]),
                        "activation_vector": hidden[0, position]
                        .to(torch.float16)
                        .tolist(),
                    }
                )
            write_json(
                shard_path,
                {
                    "case_id": str(case["case_id"]),
                    "rows": rows,
                    "baseline_continuation_token_ids": continuation_ids,
                    "baseline_continuation_text": tokenizer.decode(
                        continuation_ids, skip_special_tokens=True
                    ),
                },
            )
    finally:
        del model, tokenizer
        release_cuda_memory()
    shards = [json.loads((shard_dir / f"{case['case_id']}.json").read_text()) for case in cases]
    rows = [row for shard in shards for row in shard["rows"]]
    trajectories_path = output_root / "poetry_extract" / "trajectories.parquet"
    continuations_path = output_root / "poetry_extract" / "continuations.jsonl"
    _write_activation_parquet(trajectories_path, rows)
    write_jsonl(
        continuations_path,
        [
            {
                "case_id": shard["case_id"],
                "baseline_continuation_token_ids": shard[
                    "baseline_continuation_token_ids"
                ],
                "baseline_continuation_text": shard["baseline_continuation_text"],
            }
            for shard in shards
        ],
    )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in row["activation_vector"]
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "phase": "poetry_extract",
        "passed": len(shards) == len(cases) and finite,
        "config_sha256": config_fingerprint(config),
        "cases": len(shards),
        "positions": len(rows),
        "boundary": boundary,
        "trajectories": {
            "path": str(trajectories_path),
            "sha256": sha256_file(trajectories_path),
        },
        "continuations": {
            "path": str(continuations_path),
            "sha256": sha256_file(continuations_path),
        },
    }
    write_json(report_path, report)
    return report


def _description_shard_name(case_id: str, position: int, variant: str) -> str:
    return f"{case_id}--{position:05d}--{variant}.json"


def run_describe(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import torch

    from eval_nano_av_ar_roundtrip_gate import parse_generated_explanation
    from nano_av_warmstart_smoke import load_av_config, resolve_injection_scale
    from nano_realdata_stage3_build import DEFAULT_ACTOR_TEMPLATE
    from nla.schema import INJECT_PLACEHOLDER

    config = load_poetry_config(config_path)
    output_root, reports = _paths(config, config_path)
    report_path = reports / "poetry_describe_report.json"
    if reusable := _report_reusable(report_path, config, force=force):
        return reusable
    _require_report(reports / "poetry_extract_report.json", label="poetry extract")
    trajectories = pq.read_table(
        output_root / "poetry_extract" / "trajectories.parquet"
    ).to_pylist()
    evaluation = config["evaluation"]
    checkpoint = resolve_path(
        config["models"]["av_hf_staging"], config_path=config_path
    )
    model, tokenizer = load_av_model(
        checkpoint,
        torch_dtype=str(evaluation["torch_dtype"]),
        device_map=str(evaluation.get("av_device_map", "auto")),
    )
    av_cfg = load_av_config(
        resolve_path(config["paths"]["validation_parquet"], config_path=config_path),
        tokenizer,
    )
    injection_scale = resolve_injection_scale(
        evaluation["injection_scale"], av_cfg.d_model
    )
    prompt = [
        {
            "role": "user",
            "content": DEFAULT_ACTOR_TEMPLATE.format(
                injection_char=INJECT_PLACEHOLDER
            ),
        }
    ]
    by_key = {
        (str(row["case_id"]), int(row["relative_offset"])): row
        for row in trajectories
    }
    case_ids = sorted({str(row["case_id"]) for row in trajectories})
    shuffled_case = {
        case_id: case_ids[(index + 1) % len(case_ids)]
        for index, case_id in enumerate(case_ids)
    }
    shard_dir = output_root / "poetry_describe" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in shard_dir.glob("*.json"):
            path.unlink()
    real_samples = int(evaluation["av_samples_per_position"])
    control_samples = int(evaluation["shuffled_samples_per_position"])
    if real_samples < 1 or control_samples < 1:
        raise ObservatoryConfigError("poetry AV sample counts must be positive")
    try:
        for ordinal, row in enumerate(trajectories, start=1):
            for variant, sample_count in (
                ("real", real_samples),
                ("shuffled", control_samples),
            ):
                shard_path = shard_dir / _description_shard_name(
                    str(row["case_id"]), int(row["position"]), variant
                )
                if shard_path.is_file():
                    continue
                vector_row = row
                if variant == "shuffled":
                    vector_row = by_key[
                        (
                            shuffled_case[str(row["case_id"])],
                            int(row["relative_offset"]),
                        )
                    ]
                print(
                    f"[observatory:poetry] describe {ordinal}/{len(trajectories)} "
                    f"case={row['case_id']} pos={row['position']} {variant}",
                    flush=True,
                )
                seeds = [
                    stable_int(
                        int(evaluation["seed"]),
                        row["case_id"],
                        row["position"],
                        variant,
                        sample,
                    )
                    for sample in range(sample_count)
                ]
                generated = sample_generate_batch_full_prefix(
                    model,
                    tokenizer,
                    av_cfg,
                    {"prompt": prompt},
                    torch.tensor(
                        vector_row["activation_vector"], dtype=torch.float32
                    ),
                    seeds=seeds,
                    injection_scale=injection_scale,
                    max_new_tokens=int(evaluation["av_max_new_tokens"]),
                    temperature=float(evaluation["sampling_temperature"]),
                    top_p=float(evaluation["sampling_top_p"]),
                    stop_text="</explanation>",
                )
                records = []
                for sample_index, result in enumerate(generated):
                    records.append(
                        {
                            "case_id": str(row["case_id"]),
                            "position": int(row["position"]),
                            "relative_offset": int(row["relative_offset"]),
                            "token_text": str(row["token_text"]),
                            "variant": variant,
                            "sample_index": sample_index,
                            "source_case_id": str(vector_row["case_id"]),
                            "generated": result,
                            "parsed": parse_generated_explanation(
                                str(result["text"]), fallback="empty"
                            ),
                        }
                    )
                write_json(shard_path, {"records": records})
    finally:
        del model, tokenizer
        release_cuda_memory()
    expected = len(trajectories) * 2
    shards = sorted(shard_dir.glob("*.json"))
    if len(shards) != expected:
        raise ObservatoryConfigError(
            f"poetry descriptions expected {expected} shards, found {len(shards)}"
        )
    records = [
        record
        for path in shards
        for record in json.loads(path.read_text())["records"]
    ]
    records.sort(
        key=lambda row: (
            str(row["case_id"]),
            int(row["position"]),
            str(row["variant"]),
            int(row["sample_index"]),
        )
    )
    records_path = output_root / "poetry_describe" / "descriptions.jsonl"
    write_jsonl(records_path, records)
    usable_fraction = float(
        np.mean([bool(record["parsed"]["usable"]) for record in records])
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "phase": "poetry_describe",
        "passed": usable_fraction
        >= float(config["gates"]["minimum_usable_fraction"]),
        "config_sha256": config_fingerprint(config),
        "records": len(records),
        "usable_fraction": usable_fraction,
        "descriptions": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
        },
    }
    write_json(report_path, report)
    return report


def run_score(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    config = load_poetry_config(config_path)
    output_root, reports = _paths(config, config_path)
    report_path = reports / "poetry_score_report.json"
    if reusable := _report_reusable(report_path, config, force=force):
        return reusable
    _require_report(reports / "poetry_describe_report.json", label="poetry describe")
    cases = {
        str(case["case_id"]): case
        for case in read_jsonl(output_root / "poetry_corpus" / "cases.jsonl")
    }
    descriptions = read_jsonl(
        output_root / "poetry_describe" / "descriptions.jsonl"
    )
    continuations = {
        str(row["case_id"]): row
        for row in read_jsonl(
            output_root / "poetry_extract" / "continuations.jsonl"
        )
    }
    sample_scores: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in descriptions:
        case = cases[str(record["case_id"])]
        explanation = str(record.get("parsed", {}).get("explanation") or "")
        score = {
            "case_id": str(record["case_id"]),
            "position": int(record["position"]),
            "relative_offset": int(record["relative_offset"]),
            "variant": str(record["variant"]),
            "sample_index": int(record["sample_index"]),
            "usable": bool(record.get("parsed", {}).get("usable")),
            "target_exact": contains_term(explanation, [case["target_word"]]),
            "target_family": contains_term(explanation, case["target_terms"]),
            "alternate_family": contains_term(explanation, case["alternate_terms"]),
            "explanation": explanation,
        }
        sample_scores.append(score)
        grouped.setdefault(
            (score["case_id"], score["position"], score["variant"]), []
        ).append(score)
    position_scores: list[dict[str, Any]] = []
    for (case_id, position, variant), rows in sorted(grouped.items()):
        usable_rows = [row for row in rows if row["usable"]]

        def usable_rate(metric: str) -> float:
            if not usable_rows:
                return 0.0
            return float(np.mean([bool(row[metric]) for row in usable_rows]))

        position_scores.append(
            {
                "case_id": case_id,
                "position": position,
                "relative_offset": int(rows[0]["relative_offset"]),
                "variant": variant,
                "samples": len(rows),
                "usable_rate": float(np.mean([row["usable"] for row in rows])),
                "target_exact_rate": usable_rate("target_exact"),
                "target_family_rate": usable_rate("target_family"),
                "alternate_family_rate": usable_rate("alternate_family"),
            }
        )
    by_case_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in position_scores:
        by_case_variant.setdefault((row["case_id"], row["variant"]), []).append(row)
    onset_rate = float(config["gates"]["planning_onset_rate"])
    case_scores: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        real = sorted(
            by_case_variant[(case_id, "real")],
            key=lambda row: int(row["position"]),
        )
        shuffled = {
            int(row["position"]): row
            for row in by_case_variant[(case_id, "shuffled")]
        }
        onset = next(
            (
                row
                for row in real
                if float(row["target_family_rate"]) >= onset_rate
                and float(row["target_family_rate"])
                > float(shuffled[int(row["position"])]["target_family_rate"])
            ),
            None,
        )
        anchor = real[-1]
        anchor_control = shuffled[int(anchor["position"])]
        continuation = continuations[case_id]["baseline_continuation_text"]
        case_scores.append(
            {
                "case_id": case_id,
                "target_word": case["target_word"],
                "baseline_continuation": continuation,
                "baseline_hits_target_family": contains_term(
                    continuation, case["target_terms"]
                ),
                "planning_onset_position": (
                    None if onset is None else int(onset["position"])
                ),
                "planning_onset_relative_offset": (
                    None if onset is None else int(onset["relative_offset"])
                ),
                "anchor_real_target_family_rate": float(
                    anchor["target_family_rate"]
                ),
                "anchor_shuffled_target_family_rate": float(
                    anchor_control["target_family_rate"]
                ),
                "anchor_lift": float(anchor["target_family_rate"])
                - float(anchor_control["target_family_rate"]),
            }
        )
    score_dir = output_root / "poetry_score"
    samples_path = score_dir / "sample_scores.jsonl"
    positions_path = score_dir / "position_scores.jsonl"
    cases_path = score_dir / "case_scores.jsonl"
    write_jsonl(samples_path, sample_scores)
    write_jsonl(positions_path, position_scores)
    write_jsonl(cases_path, case_scores)
    report = {
        "schema_version": REPORT_SCHEMA,
        "phase": "poetry_score",
        "passed": len(case_scores) == len(cases),
        "config_sha256": config_fingerprint(config),
        "cases": len(case_scores),
        "cases_with_planning_onset": sum(
            row["planning_onset_position"] is not None for row in case_scores
        ),
        "cases_with_baseline_target_rhyme": sum(
            row["baseline_hits_target_family"] for row in case_scores
        ),
        "mean_anchor_lift": float(np.mean([row["anchor_lift"] for row in case_scores])),
        "artifacts": {
            "sample_scores": str(samples_path),
            "position_scores": str(positions_path),
            "case_scores": str(cases_path),
        },
    }
    write_json(report_path, report)
    return report


def run_reconstruct(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from eval_nano_ar_miles_checkpoint import (
        _load_model_and_tokenizer,
        _resolve_hf_dir,
        _sidecar_template,
        predict_prompts,
    )
    from eval_nano_av_ar_roundtrip_gate import format_critic_prompt

    config = load_poetry_config(config_path)
    output_root, reports = _paths(config, config_path)
    report_path = reports / "poetry_reconstruct_report.json"
    if reusable := _report_reusable(report_path, config, force=force):
        return reusable
    _require_report(reports / "poetry_score_report.json", label="poetry score")
    cases = {
        str(case["case_id"]): case
        for case in read_jsonl(output_root / "poetry_corpus" / "cases.jsonl")
    }
    samples = read_jsonl(output_root / "poetry_score" / "sample_scores.jsonl")
    trajectories = pq.read_table(
        output_root / "poetry_extract" / "trajectories.parquet"
    ).to_pylist()
    anchors = {
        str(row["case_id"]): row
        for row in trajectories
        if int(row["relative_offset"]) == 0
    }
    selected: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        candidates = [
            row
            for row in samples
            if row["case_id"] == case_id
            and row["variant"] == "real"
            and int(row["relative_offset"]) == 0
            and row["usable"]
        ]
        candidates.sort(
            key=lambda row: (
                not bool(row["target_family"]),
                not bool(row["target_exact"]),
                int(row["sample_index"]),
            )
        )
        if not candidates:
            continue
        original = str(candidates[0]["explanation"])
        edited, changed = edit_explanation(original, case["edit_map"])
        if not changed or edited == original:
            continue
        selected.append(
            {
                "case_id": case_id,
                "original_explanation": original,
                "edited_explanation": edited,
                "changed_terms": changed,
                "gold": np.asarray(
                    anchors[case_id]["activation_vector"], dtype=np.float32
                ),
            }
        )
    minimum_editable = int(config["gates"]["minimum_editable_cases"])
    if len(selected) < minimum_editable:
        raise ObservatoryConfigError(
            f"only {len(selected)} poetry cases are editable; minimum is {minimum_editable}"
        )
    predictions: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    for critic in [str(value) for value in evaluation.get("critics", ["primary"])]:
        model_key = f"{critic}_ar"
        if model_key not in config["models"]:
            raise ObservatoryConfigError(f"missing poetry critic model: {model_key}")
        checkpoint = resolve_path(
            config["models"][model_key], config_path=config_path
        )
        hf_dir = _resolve_hf_dir(checkpoint)
        template = _sidecar_template(
            hf_dir,
            resolve_path(
                config["paths"]["validation_parquet"], config_path=config_path
            ),
            resolve_path(config["paths"]["train_parquet"], config_path=config_path),
        )
        model, tokenizer = _load_model_and_tokenizer(
            hf_dir,
            torch_dtype=str(evaluation["torch_dtype"]),
            device_map=str(evaluation.get("ar_device_map", "auto")),
        )
        prompts = []
        for row in selected:
            prompts.extend(
                [
                    format_critic_prompt(template, row["original_explanation"]),
                    format_critic_prompt(template, row["edited_explanation"]),
                ]
            )
        prediction = predict_prompts(
            model,
            tokenizer,
            prompts,
            batch_size=int(evaluation["ar_batch_size"]),
            max_length=int(evaluation["ar_max_length"]),
        )
        predictions[critic] = np.asarray(prediction, dtype=np.float32).reshape(
            len(selected), 2, 2688
        )
        del model, tokenizer
        release_cuda_memory()
    gold = np.stack([row["gold"] for row in selected])
    primary = predictions["primary"]
    original_metrics = rowwise_reconstruction_metrics(primary[:, 0], gold)
    reconstruction_dir = output_root / "poetry_reconstruct"
    vectors_path = reconstruction_dir / "reconstructions.npz"
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vectors_path, gold=gold, **predictions)
    for index, row in enumerate(selected):
        metadata.append(
            {
                "case_id": row["case_id"],
                "vector_index": index,
                "original_explanation": row["original_explanation"],
                "edited_explanation": row["edited_explanation"],
                "changed_terms": row["changed_terms"],
                "original_directional_mse": float(
                    original_metrics["directional_mse"][index]
                ),
                "original_cosine": float(original_metrics["cosine"][index]),
                "edit_delta_norm": float(
                    np.linalg.norm(primary[index, 1] - primary[index, 0])
                ),
            }
        )
    metadata_path = reconstruction_dir / "reconstructions.jsonl"
    write_jsonl(metadata_path, metadata)
    report = {
        "schema_version": REPORT_SCHEMA,
        "phase": "poetry_reconstruct",
        "passed": len(metadata) >= minimum_editable
        and all(row["edit_delta_norm"] > 0 for row in metadata),
        "config_sha256": config_fingerprint(config),
        "eligible_cases": len(metadata),
        "critics": sorted(predictions),
        "mean_original_directional_mse": float(
            np.mean(original_metrics["directional_mse"])
        ),
        "vectors": {"path": str(vectors_path), "sha256": sha256_file(vectors_path)},
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
    }
    write_json(report_path, report)
    return report


def run_intervene(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import torch
    from transformers import PreTrainedTokenizerFast

    from nano_introspection import resolve_nano_module_paths
    from nano_r33_functional_runtime import load_target_model

    config = load_poetry_config(config_path)
    output_root, reports = _paths(config, config_path)
    report_path = reports / "poetry_intervene_report.json"
    if reusable := _report_reusable(report_path, config, force=force):
        return reusable
    _require_report(
        reports / "poetry_reconstruct_report.json", label="poetry reconstruct"
    )
    cases = {
        str(case["case_id"]): case
        for case in read_jsonl(output_root / "poetry_corpus" / "cases.jsonl")
    }
    metadata = read_jsonl(
        output_root / "poetry_reconstruct" / "reconstructions.jsonl"
    )
    with np.load(
        output_root / "poetry_reconstruct" / "reconstructions.npz",
        allow_pickle=False,
    ) as cache:
        gold = np.asarray(cache["gold"], dtype=np.float32)
        primary = np.asarray(cache["primary"], dtype=np.float32)
    prepared = {
        str(case["case_id"]): case
        for case in read_jsonl(output_root / "poetry_corpus" / "cases.jsonl")
    }
    evaluation = config["evaluation"]
    base_hf = resolve_path(config["models"]["base_hf"], config_path=config_path)
    args = SimpleNamespace(
        target_model=str(base_hf),
        target_torch_dtype=str(evaluation["torch_dtype"]),
        target_trust_remote_code=True,
        target_local_files_only=True,
        target_revision=None,
        target_device_map=str(evaluation.get("base_device_map", "auto")),
    )
    model = load_target_model(args)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_hf, trust_remote_code=True, local_files_only=True
    )
    layers = resolve_nano_module_paths(model)["layers"].obj
    boundary = int(evaluation["boundary"])
    if layers is None or not 1 <= boundary <= len(layers):
        raise ObservatoryConfigError(f"invalid poetry boundary: {boundary}")
    boundary_module = layers[boundary - 1]
    doses = [float(value) for value in evaluation["steering_doses"]]
    records: list[dict[str, Any]] = []
    try:
        for row in metadata:
            index = int(row["vector_index"])
            case_id = str(row["case_id"])
            case = cases[case_id]
            prefix = [int(value) for value in prepared[case_id]["prefix_token_ids"]]
            edit_delta = primary[index, 1] - primary[index, 0]
            generator = np.random.default_rng(
                stable_int(int(evaluation["seed"]), case_id, "random-direction")
            )
            random_delta = generator.standard_normal(edit_delta.shape).astype(np.float32)
            for direction_name, delta in (
                ("edited", edit_delta),
                ("random", random_delta),
            ):
                for dose in doses:
                    replacement = steering_replacement(gold[index], delta, dose)
                    continuation_ids = greedy_generate_patched_full_prefix(
                        model=model,
                        boundary_module=boundary_module,
                        prefix=prefix,
                        replacement=torch.tensor(replacement, dtype=torch.float32),
                        max_new_tokens=int(evaluation["continuation_tokens"]),
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    continuation = tokenizer.decode(
                        continuation_ids, skip_special_tokens=True
                    )
                    records.append(
                        {
                            "case_id": case_id,
                            "direction": direction_name,
                            "dose": dose,
                            "continuation_token_ids": continuation_ids,
                            "continuation_text": continuation,
                            "hits_target_family": contains_term(
                                continuation, case["target_terms"]
                            ),
                            "hits_alternate_family": contains_term(
                                continuation, case["alternate_terms"]
                            ),
                        }
                    )
    finally:
        del model, tokenizer
        release_cuda_memory()
    records_path = output_root / "poetry_intervene" / "interventions.jsonl"
    write_jsonl(records_path, records)
    edited = [row for row in records if row["direction"] == "edited"]
    random = [row for row in records if row["direction"] == "random"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "phase": "poetry_intervene",
        "passed": len(records) == len(metadata) * len(doses) * 2,
        "config_sha256": config_fingerprint(config),
        "cases": len(metadata),
        "records": len(records),
        "steering_doses": doses,
        "edited_alternate_hit_rate": float(
            np.mean([row["hits_alternate_family"] for row in edited])
        ),
        "random_alternate_hit_rate": float(
            np.mean([row["hits_alternate_family"] for row in random])
        ),
        "interventions": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
        },
    }
    write_json(report_path, report)
    return report


PHASE_RUNNERS = {
    "poetry-prepare": run_prepare,
    "poetry-extract": run_extract,
    "poetry-describe": run_describe,
    "poetry-score": run_score,
    "poetry-reconstruct": run_reconstruct,
    "poetry-intervene": run_intervene,
}
