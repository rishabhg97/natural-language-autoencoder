#!/usr/bin/env python3
"""Run fail-closed, resumable GPU phases for the offline NLA Observatory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    read_jsonl,
    resolve_path,
    sha256_file,
    write_json,
)
from .model_runtime import (
    baseline_wake_logits,
    compare_score_batches,
    control_vectors,
    functional_wake_metrics,
    greedy_generate_patched_cached,
    greedy_generate_patched_full_prefix,
    greedy_generate_unpatched,
    hf_checkpoint_complete,
    load_av_model,
    load_train_mean,
    read_parquet_rows,
    release_cuda_memory,
    rowwise_reconstruction_metrics,
    run_functional_pass_detailed,
    sample_generate_batch_full_prefix,
    select_trajectory_positions,
    selected_rows,
    teacher_forced_scores,
    write_prediction_parquet,
    write_trajectory_parquet,
)


SCHEMA_VERSION = "nano_viz_model_batches.v1"


def _configure_runtime() -> None:
    os.environ.setdefault("NLA_TRAIN_MAMBA_KERNEL_MODE", "unfused_torch_conv")
    os.environ.setdefault("NLA_CRITIC_FWD_DISABLE_MAMBA_FAST_PATH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _reusable_report(
    report_path: Path, config: dict[str, Any], *, force: bool
) -> dict[str, Any] | None:
    if force or not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    if not report.get("passed"):
        return None
    if report.get("config_sha256") != config_fingerprint(config):
        return None
    return report


def _canary_rows(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    paths = config["paths"]
    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    manifest = json.loads((corpus_dir / "selection_manifest.json").read_text())
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    return selected_rows(source_rows, list(manifest["canary_row_ids"]))


def _validation_rows(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    import numpy as np

    paths = config["paths"]
    all_rows = read_parquet_rows(
        resolve_path(paths["validation_parquet"], config_path=config_path)
    )
    generated_rows = read_jsonl(
        resolve_path(paths["generated_validation_jsonl"], config_path=config_path)
    )
    cache_path = resolve_path(
        paths["validation_prediction_cache_npz"], config_path=config_path
    )
    with np.load(cache_path, allow_pickle=False) as cache:
        row_indices = np.asarray(cache["validation__row_indices"], dtype=np.int64)
        doc_ids = np.asarray(cache["validation__doc_ids"], dtype=np.str_)
        family_ids = np.asarray(
            cache["validation__content_family_ids"], dtype=np.str_
        )
    if not (
        len(row_indices) == len(doc_ids) == len(family_ids) == len(generated_rows)
    ):
        raise ObservatoryConfigError(
            "qualified validation evidence has inconsistent row counts: "
            f"cache={len(row_indices)} generated={len(generated_rows)}"
        )
    generated_by_index = {
        int(record["row_index"]): record for record in generated_rows
    }
    if len(generated_by_index) != len(generated_rows):
        raise ObservatoryConfigError(
            "generated validation evidence contains duplicate row indices"
        )
    rows: list[dict[str, Any]] = []
    for cache_index, row_index in enumerate(row_indices):
        evidence = generated_by_index.get(int(row_index))
        if evidence is None:
            raise ObservatoryConfigError(
                f"qualified cache row {int(row_index)} is missing generated evidence"
            )
        if str(evidence.get("split") or "") != "validation":
            raise ObservatoryConfigError(
                f"qualified cache row {int(row_index)} is not validation evidence"
            )
        source_index = int(evidence["source_row_index"])
        if not 0 <= source_index < len(all_rows):
            raise ObservatoryConfigError(
                f"source_row_index out of bounds for row {int(row_index)}: {source_index}"
            )
        row = dict(all_rows[source_index])
        expected_doc = str(doc_ids[cache_index])
        if (
            str(row.get("doc_id") or "") != expected_doc
            or str(evidence.get("doc_id") or "") != expected_doc
        ):
            raise ObservatoryConfigError(
                f"validation cache doc identity mismatch at row {int(row_index)}"
            )
        expected_tokens = int(evidence["n_raw_tokens"])
        if int(row.get("n_raw_tokens", -1)) != expected_tokens:
            raise ObservatoryConfigError(
                f"validation token identity mismatch at row {int(row_index)}"
            )
        row["source_row_index"] = source_index
        row["row_index"] = int(row_index)
        row["content_family_id"] = str(family_ids[cache_index])
        row["split"] = "validation"
        rows.append(row)
    return rows


def _resolved_lattice_cells(
    interventions: list[dict[str, Any]],
    alternate_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind model-generated tellings into the immutable intervention registry."""

    alternates = {str(record["cell_id"]): record for record in alternate_records}
    if len(alternates) != len(alternate_records):
        raise ObservatoryConfigError("alternate-telling records contain duplicate cell ids")
    resolved: list[dict[str, Any]] = []
    used_alternates: set[str] = set()
    for source_cell in interventions:
        cell = dict(source_cell)
        if cell.get("family") == "alternate_telling":
            cell_id = str(cell["cell_id"])
            record = alternates.get(cell_id)
            if record is None:
                raise ObservatoryConfigError(
                    f"alternate-telling cell is missing generated text: {cell_id}"
                )
            parsed = record.get("parsed") or {}
            explanation = str(parsed.get("explanation") or "").strip()
            if not parsed.get("usable") or not explanation:
                raise ObservatoryConfigError(
                    f"alternate-telling cell is not parse-usable: {cell_id}"
                )
            cell["text"] = explanation
            cell["state"] = "ready"
            used_alternates.add(cell_id)
        if cell.get("state") != "ready" or not str(cell.get("text") or "").strip():
            raise ObservatoryConfigError(
                f"lattice cell is not ready for AR encoding: {cell.get('cell_id')}"
            )
        resolved.append(cell)
    unused = sorted(set(alternates) - used_alternates)
    if unused:
        raise ObservatoryConfigError(
            f"alternate-telling output contains unknown cell ids: {unused[:5]}"
        )
    cell_ids = [str(cell["cell_id"]) for cell in resolved]
    if len(cell_ids) != len(set(cell_ids)):
        raise ObservatoryConfigError("resolved lattice contains duplicate cell ids")
    return resolved


def _prediction_shard_valid(
    path: Path, *, expected_cell_ids: list[str], critic: str
) -> bool:
    if not path.is_file():
        return False
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["cell_id", "critic"])
        data = table.to_pydict()
    except (OSError, ValueError):
        return False
    return data["cell_id"] == expected_cell_ids and data["critic"] == [critic] * len(
        expected_cell_ids
    )


def _score_controls(
    model: Any,
    tokenizer: Any,
    av_cfg: Any,
    rows: list[dict[str, Any]],
    controls: list[dict[str, Any | None]],
    *,
    names: list[str],
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    flattened_rows: list[dict[str, Any]] = []
    flattened_vectors: list[Any | None] = []
    variants: list[str] = []
    for row, row_controls in zip(rows, controls, strict=True):
        for name in names:
            flattened_rows.append(row)
            flattened_vectors.append(row_controls[name])
            variants.append(name)
    scored = teacher_forced_scores(
        model,
        tokenizer,
        av_cfg,
        flattened_rows,
        flattened_vectors,
        injection_scale=float(evaluation["injection_scale"]),
        max_target_tokens=int(evaluation["av_max_target_tokens"]),
        batch_size=int(evaluation["av_score_batch_size"]),
    )
    for record, variant in zip(scored, variants, strict=True):
        record["variant"] = variant
    return scored


def run_canary_av(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np

    from eval_nano_av_ar_roundtrip_gate import parse_generated_explanation
    from nano_av_warmstart_smoke import (
        generate_controls_for_row,
        load_av_config,
        resolve_injection_scale,
    )

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    output_dir = (
        resolve_path(paths["model_outputs_dir"], config_path=config_path) / "e1_canary"
    )
    report_path = output_dir / "canary_av_report.json"
    records_path = output_dir / "canary_av_records.jsonl"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _canary_rows(config, config_path)
    if len(rows) != int(config["selection"]["canary_rows"]):
        raise ObservatoryConfigError(f"expected 8 canary rows, found {len(rows)}")
    train_mean = load_train_mean(
        resolve_path(paths["validation_prediction_cache_npz"], config_path=config_path)
    )
    controls = control_vectors(rows, train_mean)
    checkpoint = resolve_path(models["av_hf_staging"], config_path=config_path)
    model, tokenizer = load_av_model(
        checkpoint,
        torch_dtype=str(evaluation["torch_dtype"]),
        device_map=str(evaluation.get("av_device_map", "auto")),
    )
    av_cfg = load_av_config(
        resolve_path(paths["validation_parquet"], config_path=config_path), tokenizer
    )
    injection_scale = resolve_injection_scale(
        evaluation["injection_scale"], av_cfg.d_model
    )
    control_names = [str(value) for value in evaluation["controls"]]
    scores = _score_controls(
        model,
        tokenizer,
        av_cfg,
        rows,
        controls,
        names=control_names,
        evaluation=evaluation,
    )
    real_vectors = [row["real"] for row in controls]
    candidate_batches = sorted(
        {
            1,
            *(
                int(value)
                for value in evaluation.get(
                    "av_score_candidate_batches", [1, 2, 4, len(rows)]
                )
            ),
        }
    )
    if any(batch <= 0 or batch > len(rows) for batch in candidate_batches):
        raise ObservatoryConfigError(
            f"AV score candidate batches must be in [1, {len(rows)}]"
        )
    scores_by_batch: dict[int, list[dict[str, Any]]] = {}
    for batch in candidate_batches:
        scores_by_batch[batch] = teacher_forced_scores(
            model,
            tokenizer,
            av_cfg,
            rows,
            real_vectors,
            injection_scale=injection_scale,
            max_target_tokens=int(evaluation["av_max_target_tokens"]),
            batch_size=batch,
        )
    score_batch1 = scores_by_batch[1]
    equivalence_candidates: dict[str, Any] = {}
    for batch in candidate_batches:
        comparison = compare_score_batches(score_batch1, scores_by_batch[batch])
        comparison["qualified"] = (
            comparison["max_abs_loss_delta"]
            <= float(gates["av_batch_equivalence_max_loss_delta"])
            and comparison["max_abs_token_logprob_delta"]
            <= float(gates["av_batch_equivalence_max_logprob_delta"])
        )
        equivalence_candidates[str(batch)] = comparison
    qualified_batches = [
        int(batch)
        for batch, comparison in equivalence_candidates.items()
        if comparison["qualified"]
    ]
    selected_batch = max(qualified_batches)
    configured_batch = int(evaluation["av_score_batch_size"])
    if configured_batch not in candidate_batches:
        raise ObservatoryConfigError(
            "av_score_batch_size must be included in av_score_candidate_batches"
        )
    equivalence = equivalence_candidates[str(configured_batch)]
    existing_records = read_jsonl(records_path) if records_path.is_file() and not force else []
    existing_by_row = {
        int(record["row_index"]): record for record in existing_records
    }
    reuse_generation = (
        len(existing_by_row) == len(rows)
        and all(
            int(row["row_index"]) in existing_by_row
            and all(
                name in (existing_by_row[int(row["row_index"])].get("controls") or {})
                and (
                    existing_by_row[int(row["row_index"])]
                    ["controls"][name]
                    .get("parsed")
                    is not None
                )
                for name in control_names
            )
            for row in rows
        )
    )
    records: list[dict[str, Any]] = []
    if reuse_generation:
        records = [existing_by_row[int(row["row_index"])] for row in rows]
    else:
        for ordinal, (row, row_controls) in enumerate(
            zip(rows, controls, strict=True), start=1
        ):
            print(
                f"[observatory:e1] generation row {ordinal}/{len(rows)} "
                f"row_index={row['row_index']}",
                flush=True,
            )
            generated = generate_controls_for_row(
                model,
                tokenizer,
                av_cfg,
                row,
                {name: row_controls[name] for name in control_names},
                control_names,
                injection_scale=injection_scale,
                max_new_tokens=int(evaluation["max_new_tokens"]),
                generation_prefix="",
                stop_text="</explanation>",
                use_cache=False,
                batch_full_prefix=True,
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_index": int(row["row_index"]),
                    "doc_id": str(row["doc_id"]),
                    "content_family_id": str(row["content_family_id"]),
                    "controls": {
                        name: {
                            "generated": generated[name],
                            "parsed": parse_generated_explanation(
                                generated[name], fallback="empty"
                            ),
                        }
                        for name in control_names
                    },
                }
            )
    del model, tokenizer
    release_cuda_memory()
    with records_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    score_path = output_dir / "canary_token_logprobs.jsonl"
    with score_path.open("w") as handle:
        for record in scores:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    equivalence_paths: dict[str, dict[str, str]] = {}
    for batch, values in scores_by_batch.items():
        name = f"batch{batch}"
        path = output_dir / f"canary_real_{name}_token_logprobs.jsonl"
        with path.open("w") as handle:
            for record in values:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        equivalence_paths[name] = {"path": str(path), "sha256": sha256_file(path)}
    parse = {
        name: {
            "closed_fraction": float(
                np.mean(
                    [
                        bool(record["controls"][name]["parsed"]["closed"])
                        for record in records
                    ]
                )
            ),
            "usable_fraction": float(
                np.mean(
                    [
                        bool(record["controls"][name]["parsed"]["usable"])
                        for record in records
                    ]
                )
            ),
        }
        for name in control_names
    }
    losses = {
        name: float(np.mean([item["loss"] for item in scores if item["variant"] == name]))
        for name in control_names
    }
    passed = (
        parse["real"]["usable_fraction"]
        >= float(gates["require_parse_usable_fraction"])
        and parse["real"]["closed_fraction"]
        >= float(gates["require_parse_closed_fraction"])
        and bool(equivalence["qualified"])
        and all(np.isfinite(item["loss"]) for item in scores)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "e1_canary_av",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "rows": len(rows),
        "controls": control_names,
        "losses": losses,
        "parse": parse,
        "batch_equivalence": equivalence,
        "batch_equivalence_candidates": equivalence_candidates,
        "batch_equivalence_records": equivalence_paths,
        "batch_selection": {
            "configured": configured_batch,
            "largest_qualified": selected_batch,
            "candidate_batches": candidate_batches,
        },
        "generation_reused": reuse_generation,
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "token_logprobs": {"path": str(score_path), "sha256": sha256_file(score_path)},
    }
    write_json(report_path, report)
    return report


def run_token_logprobs(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np

    from nano_av_warmstart_smoke import load_av_config, resolve_injection_scale

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    output_dir = (
        resolve_path(paths["model_outputs_dir"], config_path=config_path)
        / "e2_token_logprobs"
    )
    report_path = output_dir / "token_logprobs_report.json"
    records_path = output_dir / "validation_token_logprobs.jsonl"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    canary_report = (
        resolve_path(paths["model_outputs_dir"], config_path=config_path)
        / "e1_canary"
        / "canary_av_report.json"
    )
    if not canary_report.is_file() or not json.loads(canary_report.read_text()).get(
        "passed"
    ):
        raise ObservatoryConfigError("E1 AV canary must pass before E2")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _validation_rows(config, config_path)
    if len(rows) != 512 or any(row["split"] != "validation" for row in rows):
        raise ObservatoryConfigError(
            f"E2 is validation-only and requires exactly 512 rows; found {len(rows)}"
        )
    train_mean = load_train_mean(
        resolve_path(paths["validation_prediction_cache_npz"], config_path=config_path)
    )
    controls = control_vectors(rows, train_mean)
    checkpoint = resolve_path(models["av_hf_staging"], config_path=config_path)
    model, tokenizer = load_av_model(
        checkpoint,
        torch_dtype=str(evaluation["torch_dtype"]),
        device_map=str(evaluation.get("av_device_map", "auto")),
    )
    av_cfg = load_av_config(
        resolve_path(paths["validation_parquet"], config_path=config_path), tokenizer
    )
    injection_scale = resolve_injection_scale(
        evaluation["injection_scale"], av_cfg.d_model
    )
    control_names = [str(value) for value in evaluation["controls"]]
    flattened_rows: list[dict[str, Any]] = []
    flattened_vectors: list[Any | None] = []
    variants: list[str] = []
    for row, row_controls in zip(rows, controls, strict=True):
        for name in control_names:
            flattened_rows.append(row)
            flattened_vectors.append(row_controls[name])
            variants.append(name)
    scores = teacher_forced_scores(
        model,
        tokenizer,
        av_cfg,
        flattened_rows,
        flattened_vectors,
        injection_scale=injection_scale,
        max_target_tokens=int(evaluation["av_max_target_tokens"]),
        batch_size=int(evaluation["av_score_batch_size"]),
    )
    del model, tokenizer
    release_cuda_memory()
    with records_path.open("w") as handle:
        for score, variant in zip(scores, variants, strict=True):
            score["variant"] = variant
            handle.write(json.dumps(score, sort_keys=True) + "\n")
    losses_by_variant = {
        name: np.asarray(
            [score["loss"] for score in scores if score["variant"] == name],
            dtype=np.float64,
        )
        for name in control_names
    }
    paired = {
        name: {
            "mean_real_minus_control": float(
                np.mean(losses_by_variant["real"] - losses_by_variant[name])
            ),
            "real_win_fraction": float(
                np.mean(losses_by_variant["real"] < losses_by_variant[name])
            ),
        }
        for name in control_names
        if name != "real"
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "e2_token_logprobs",
        "passed": (
            len(scores) == len(rows) * len(control_names)
            and all(np.isfinite(score["loss"]) for score in scores)
            and len({score["row_index"] for score in scores}) == 512
        ),
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len(rows),
        "records": len(scores),
        "controls": control_names,
        "mean_loss": {
            name: float(np.mean(values)) for name, values in losses_by_variant.items()
        },
        "paired": paired,
        "token_logprobs": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
        },
    }
    write_json(report_path, report)
    return report


def _prediction_equivalence(left: Any, right: Any) -> dict[str, float]:
    import numpy as np

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    differences = left_array - right_array
    relative_l2 = np.linalg.norm(differences, axis=1) / np.maximum(
        np.linalg.norm(left_array, axis=1), 1e-12
    )
    cosine = np.sum(left_array * right_array, axis=1) / np.maximum(
        np.linalg.norm(left_array, axis=1) * np.linalg.norm(right_array, axis=1),
        1e-12,
    )
    return {
        "max_relative_l2": float(relative_l2.max()),
        "mean_relative_l2": float(relative_l2.mean()),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
    }


def run_canary_ar(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np

    from eval_nano_ar_miles_checkpoint import (
        _load_model_and_tokenizer,
        _resolve_hf_dir,
        _sidecar_template,
        predict_prompts,
    )
    from eval_nano_av_ar_roundtrip_gate import format_critic_prompt
    from nano_eval_core import activation_reconstruction_metrics

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model_outputs = resolve_path(
        paths["model_outputs_dir"], config_path=config_path
    )
    output_dir = model_outputs / "e1_canary"
    report_path = output_dir / "canary_ar_report.json"
    vectors_path = output_dir / "canary_ar_predictions.npz"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    av_report_path = output_dir / "canary_av_report.json"
    av_records_path = output_dir / "canary_av_records.jsonl"
    if not av_report_path.is_file() or not json.loads(av_report_path.read_text()).get(
        "passed"
    ):
        raise ObservatoryConfigError("E1 AV canary must pass before AR scoring")
    rows = _canary_rows(config, config_path)
    generated = {
        int(record["row_index"]): record for record in read_jsonl(av_records_path)
    }
    control_names = [str(value) for value in evaluation["controls"]]
    variants = ["teacher", *[f"av_{name}" for name in control_names]]
    explanations: dict[str, list[str]] = {"teacher": []}
    explanations.update({f"av_{name}": [] for name in control_names})
    for row in rows:
        record = generated.get(int(row["row_index"]))
        if record is None:
            raise ObservatoryConfigError(
                f"missing AV canary generation for row {row['row_index']}"
            )
        explanations["teacher"].append(str(row["api_explanation"]))
        for name in control_names:
            explanations[f"av_{name}"].append(
                str(record["controls"][name]["parsed"]["explanation"])
            )
    targets = np.asarray([row["activation_vector"] for row in rows], dtype=np.float32)
    train_mean = load_train_mean(
        resolve_path(paths["validation_prediction_cache_npz"], config_path=config_path)
    )
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    equivalence: dict[str, Any] = {}
    equivalence_candidates: dict[str, Any] = {}
    batch_selection: dict[str, Any] = {}
    critic_templates: dict[str, str] = {}
    configured_batch = int(evaluation["ar_batch_size"])
    candidate_batches = sorted(
        {
            1,
            configured_batch,
            *(
                int(value)
                for value in evaluation.get(
                    "ar_batch_candidate_batches", [1, configured_batch]
                )
            ),
        }
    )
    if any(value < 1 for value in candidate_batches):
        raise ObservatoryConfigError("AR candidate batch sizes must be positive")
    max_batch_l2 = float(gates["ar_batch_equivalence_max_relative_l2"])
    for critic_name, model_key in (
        ("primary", "primary_ar"),
        ("independent", "independent_ar"),
    ):
        checkpoint = resolve_path(models[model_key], config_path=config_path)
        hf_dir = _resolve_hf_dir(checkpoint)
        critic_template = _sidecar_template(
            hf_dir,
            resolve_path(paths["validation_parquet"], config_path=config_path),
            resolve_path(paths["train_parquet"], config_path=config_path),
        )
        critic_templates[critic_name] = critic_template
        model, tokenizer = _load_model_and_tokenizer(
            hf_dir,
            torch_dtype=str(evaluation["torch_dtype"]),
            device_map=str(evaluation.get("ar_device_map", "auto")),
        )
        for variant in variants:
            prompts = [
                format_critic_prompt(critic_template, explanation)
                for explanation in explanations[variant]
            ]
            prediction = predict_prompts(
                model,
                tokenizer,
                prompts,
                batch_size=configured_batch,
                max_length=int(evaluation["ar_max_length"]),
            )
            predictions[f"{critic_name}__{variant}"] = prediction
            result = activation_reconstruction_metrics(
                prediction, targets, train_mean=train_mean
            )
            metrics.setdefault(critic_name, {})[variant] = {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in result.items()
            }
        real_prompts = [
            format_critic_prompt(critic_template, explanation)
            for explanation in explanations["av_real"]
        ]
        batch1 = predict_prompts(
            model,
            tokenizer,
            real_prompts,
            batch_size=1,
            max_length=int(evaluation["ar_max_length"]),
        )
        critic_candidates: dict[str, Any] = {}
        for candidate_batch in candidate_batches:
            if candidate_batch == 1:
                candidate_prediction = batch1
            elif candidate_batch == configured_batch:
                candidate_prediction = predictions[f"{critic_name}__av_real"]
            else:
                candidate_prediction = predict_prompts(
                    model,
                    tokenizer,
                    real_prompts,
                    batch_size=candidate_batch,
                    max_length=int(evaluation["ar_max_length"]),
                )
            candidate_result = _prediction_equivalence(batch1, candidate_prediction)
            candidate_result["qualified"] = (
                candidate_result["max_relative_l2"] <= max_batch_l2
            )
            critic_candidates[str(candidate_batch)] = candidate_result
        selected_batch = max(
            int(batch)
            for batch, result in critic_candidates.items()
            if result["qualified"]
        )
        equivalence_candidates[critic_name] = critic_candidates
        equivalence[critic_name] = critic_candidates[str(configured_batch)]
        batch_selection[critic_name] = {
            "configured": configured_batch,
            "largest_qualified": selected_batch,
            "candidate_batches": candidate_batches,
        }
        del model, tokenizer
        release_cuda_memory()
    arrays: dict[str, Any] = {
        "row_indices": np.asarray([row["row_index"] for row in rows], dtype=np.int64),
        "content_family_ids": np.asarray(
            [row["content_family_id"] for row in rows], dtype=np.str_
        ),
        "targets": targets,
        **predictions,
    }
    with vectors_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    max_directional = float(gates["canary_ar_max_directional_mse"])
    passed = all(
        float(metrics[critic]["av_real"]["directional_mse"]) <= max_directional
        and float(metrics[critic]["av_real"]["directional_mse"])
        < float(metrics[critic]["av_mean"]["directional_mse"])
        and bool(equivalence[critic]["qualified"])
        for critic in ("primary", "independent")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "e1_canary_ar",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "rows": len(rows),
        "variants": variants,
        "metrics": metrics,
        "batch_equivalence": equivalence,
        "batch_equivalence_candidates": equivalence_candidates,
        "batch_selection": batch_selection,
        "critic_templates": critic_templates,
        "predictions": {
            "path": str(vectors_path),
            "sha256": sha256_file(vectors_path),
        },
    }
    write_json(report_path, report)
    return report


def run_lattice_pilot(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np

    from eval_nano_ar_miles_checkpoint import (
        _load_model_and_tokenizer,
        _resolve_hf_dir,
        _sidecar_template,
        predict_prompts,
    )
    from eval_nano_av_ar_roundtrip_gate import format_critic_prompt

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model_outputs = resolve_path(
        paths["model_outputs_dir"], config_path=config_path
    )
    output_dir = model_outputs / "e3_lattice_pilot"
    report_path = output_dir / "lattice_pilot_report.json"
    predictions_path = output_dir / "lattice_pilot_predictions.parquet"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    canary_report_path = model_outputs / "e1_canary" / "canary_ar_report.json"
    if not canary_report_path.is_file() or not json.loads(
        canary_report_path.read_text()
    ).get("passed"):
        raise ObservatoryConfigError("E1 twin-critic canary must pass before E3")
    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    selection = json.loads((corpus_dir / "selection_manifest.json").read_text())
    canary_ids = set(selection["canary_row_ids"])
    source_rows = _canary_rows(config, config_path)
    sources = {int(row["row_index"]): row for row in source_rows}
    cells = [
        cell
        for cell in read_jsonl(corpus_dir / "interventions.jsonl")
        if cell["row_id"] in canary_ids and cell["state"] == "ready"
    ]
    if not cells or any(cell["row_index"] not in sources for cell in cells):
        raise ObservatoryConfigError("E3 pilot cells do not bind to the canary sources")
    court_families = {"identity", "paraphrase", "corruption"}
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for critic_name, model_key in (
        ("primary", "primary_ar"),
        ("independent", "independent_ar"),
    ):
        critic_cells = (
            cells
            if critic_name == "primary"
            else [cell for cell in cells if cell["family"] in court_families]
        )
        checkpoint = resolve_path(models[model_key], config_path=config_path)
        hf_dir = _resolve_hf_dir(checkpoint)
        critic_template = _sidecar_template(
            hf_dir,
            resolve_path(paths["validation_parquet"], config_path=config_path),
            resolve_path(paths["train_parquet"], config_path=config_path),
        )
        prompts = [
            format_critic_prompt(critic_template, str(cell["text"]))
            for cell in critic_cells
        ]
        model, tokenizer = _load_model_and_tokenizer(
            hf_dir,
            torch_dtype=str(evaluation["torch_dtype"]),
            device_map=str(evaluation.get("ar_device_map", "auto")),
        )
        prediction = predict_prompts(
            model,
            tokenizer,
            prompts,
            batch_size=int(evaluation["ar_batch_size"]),
            max_length=int(evaluation["ar_max_length"]),
        )
        del model, tokenizer
        release_cuda_memory()
        targets = np.asarray(
            [sources[int(cell["row_index"])]["activation_vector"] for cell in critic_cells],
            dtype=np.float32,
        )
        metric = rowwise_reconstruction_metrics(prediction, targets)
        for index, cell in enumerate(critic_cells):
            source = sources[int(cell["row_index"])]
            output_rows.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "row_id": str(cell["row_id"]),
                    "row_index": int(cell["row_index"]),
                    "content_family_id": str(source["content_family_id"]),
                    "family": str(cell["family"]),
                    "variant": str(cell["variant"]),
                    "depth": str(cell["depth"]),
                    "critic": critic_name,
                    "directional_mse": float(metric["directional_mse"][index]),
                    "raw_mse": float(metric["raw_mse"][index]),
                    "cosine": float(metric["cosine"][index]),
                    "norm_ratio": float(metric["norm_ratio"][index]),
                    "prediction_vector": prediction[index].astype(np.float16).tolist(),
                }
            )
        summaries[critic_name] = {
            "cells": len(critic_cells),
            "mean_directional_mse": float(metric["directional_mse"].mean()),
            "mean_cosine": float(metric["cosine"].mean()),
            "identity_directional_mse": float(
                np.mean(
                    [
                        metric["directional_mse"][index]
                        for index, cell in enumerate(critic_cells)
                        if cell["family"] == "identity"
                    ]
                )
            ),
        }
    write_prediction_parquet(predictions_path, output_rows)
    passed = (
        len({cell["row_id"] for cell in cells}) == int(config["selection"]["canary_rows"])
        and all(
            summary["identity_directional_mse"]
            <= float(gates["canary_ar_max_directional_mse"])
            for summary in summaries.values()
        )
        and all(np.isfinite(row["directional_mse"]) for row in output_rows)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "e3_lattice_pilot",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len({cell["row_id"] for cell in cells}),
        "ready_cells": len(cells),
        "prediction_rows": len(output_rows),
        "summaries": summaries,
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
    }
    write_json(report_path, report)
    return report


def run_lattice_full(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    """Encode the complete validation lattice with resumable critic shards."""

    import numpy as np
    import pyarrow.parquet as pq

    from eval_nano_ar_miles_checkpoint import (
        _load_model_and_tokenizer,
        _resolve_hf_dir,
        _sidecar_template,
        predict_prompts,
    )
    from eval_nano_av_ar_roundtrip_gate import format_critic_prompt

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model_outputs = resolve_path(paths["model_outputs_dir"], config_path=config_path)
    output_dir = model_outputs / "p2_lattice_full"
    report_path = output_dir / "lattice_full_report.json"
    shard_dir = output_dir / "prediction_shards"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    pilot_report = model_outputs / "e3_lattice_pilot" / "lattice_pilot_report.json"
    tellings_report = (
        model_outputs / "p1_alternate_tellings" / "alternate_tellings_report.json"
    )
    for dependency, label in (
        (pilot_report, "E3 lattice pilot"),
        (tellings_report, "P1 alternate tellings"),
    ):
        if not dependency.is_file() or not json.loads(dependency.read_text()).get(
            "passed"
        ):
            raise ObservatoryConfigError(f"{label} must pass before the full lattice")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if force:
        report_path.unlink(missing_ok=True)
        for stale in shard_dir.glob("*.parquet"):
            stale.unlink()

    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    selection = json.loads((corpus_dir / "selection_manifest.json").read_text())
    expected_row_ids = [str(value) for value in selection["deep_dive_row_ids"]]
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    selected_source = selected_rows(source_rows, expected_row_ids)
    sources = {int(row["row_index"]): row for row in selected_source}
    alternate_records = read_jsonl(
        model_outputs / "p1_alternate_tellings" / "alternate_tellings.jsonl"
    )
    cells = _resolved_lattice_cells(
        read_jsonl(corpus_dir / "interventions.jsonl"), alternate_records
    )
    if {str(cell["row_id"]) for cell in cells} != set(expected_row_ids):
        raise ObservatoryConfigError("full lattice row identities differ from the selection")
    if any(int(cell["row_index"]) not in sources for cell in cells):
        raise ObservatoryConfigError("full lattice contains a cell without a source target")

    court_families = {"identity", "paraphrase", "corruption"}
    shard_size = int(evaluation.get("lattice_shard_cells", 512))
    if shard_size <= 0:
        raise ObservatoryConfigError("lattice_shard_cells must be positive")
    summaries: dict[str, Any] = {}
    shard_manifest: list[dict[str, Any]] = []
    all_finite = True
    predicted_cell_ids: dict[str, list[str]] = {}

    for critic_name, model_key in (
        ("primary", "primary_ar"),
        ("independent", "independent_ar"),
    ):
        critic_cells = (
            cells
            if critic_name == "primary"
            else [cell for cell in cells if cell["family"] in court_families]
        )
        shards = [
            critic_cells[start : start + shard_size]
            for start in range(0, len(critic_cells), shard_size)
        ]
        missing_shards = []
        for shard_index, shard_cells in enumerate(shards):
            path = shard_dir / f"{critic_name}_{shard_index:04d}.parquet"
            expected_ids = [str(cell["cell_id"]) for cell in shard_cells]
            if not _prediction_shard_valid(
                path, expected_cell_ids=expected_ids, critic=critic_name
            ):
                missing_shards.append((shard_index, shard_cells, path))

        if missing_shards:
            checkpoint = resolve_path(models[model_key], config_path=config_path)
            hf_dir = _resolve_hf_dir(checkpoint)
            critic_template = _sidecar_template(
                hf_dir,
                resolve_path(paths["validation_parquet"], config_path=config_path),
                resolve_path(paths["train_parquet"], config_path=config_path),
            )
            model, tokenizer = _load_model_and_tokenizer(
                hf_dir,
                torch_dtype=str(evaluation["torch_dtype"]),
                device_map=str(evaluation.get("ar_device_map", "auto")),
            )
            try:
                for ordinal, (shard_index, shard_cells, path) in enumerate(
                    missing_shards, start=1
                ):
                    print(
                        f"[observatory:p2] {critic_name} shard "
                        f"{ordinal}/{len(missing_shards)} cells={len(shard_cells)}",
                        flush=True,
                    )
                    prompts = [
                        format_critic_prompt(critic_template, str(cell["text"]))
                        for cell in shard_cells
                    ]
                    prediction = predict_prompts(
                        model,
                        tokenizer,
                        prompts,
                        batch_size=int(evaluation["ar_batch_size"]),
                        max_length=int(evaluation["ar_max_length"]),
                    )
                    targets = np.asarray(
                        [
                            sources[int(cell["row_index"])]["activation_vector"]
                            for cell in shard_cells
                        ],
                        dtype=np.float32,
                    )
                    metric = rowwise_reconstruction_metrics(prediction, targets)
                    output_rows: list[dict[str, Any]] = []
                    for index, cell in enumerate(shard_cells):
                        source = sources[int(cell["row_index"])]
                        output_rows.append(
                            {
                                "cell_id": str(cell["cell_id"]),
                                "row_id": str(cell["row_id"]),
                                "row_index": int(cell["row_index"]),
                                "content_family_id": str(source["content_family_id"]),
                                "family": str(cell["family"]),
                                "variant": str(cell["variant"]),
                                "depth": str(cell["depth"]),
                                "critic": critic_name,
                                "directional_mse": float(metric["directional_mse"][index]),
                                "raw_mse": float(metric["raw_mse"][index]),
                                "cosine": float(metric["cosine"][index]),
                                "norm_ratio": float(metric["norm_ratio"][index]),
                                "prediction_vector": prediction[index]
                                .astype(np.float16)
                                .tolist(),
                            }
                        )
                    write_prediction_parquet(path, output_rows)
            finally:
                del model, tokenizer
                release_cuda_memory()

        critic_cell_ids: list[str] = []
        directional_mse: list[float] = []
        cosine: list[float] = []
        identity_directional_mse: list[float] = []
        for shard_index, shard_cells in enumerate(shards):
            path = shard_dir / f"{critic_name}_{shard_index:04d}.parquet"
            expected_ids = [str(cell["cell_id"]) for cell in shard_cells]
            if not _prediction_shard_valid(
                path, expected_cell_ids=expected_ids, critic=critic_name
            ):
                raise ObservatoryConfigError(f"invalid lattice prediction shard: {path}")
            table = pq.read_table(
                path,
                columns=["cell_id", "family", "directional_mse", "cosine", "norm_ratio"],
            ).to_pydict()
            critic_cell_ids.extend(str(value) for value in table["cell_id"])
            directional_mse.extend(float(value) for value in table["directional_mse"])
            cosine.extend(float(value) for value in table["cosine"])
            identity_directional_mse.extend(
                float(value)
                for family, value in zip(
                    table["family"], table["directional_mse"], strict=True
                )
                if family == "identity"
            )
            all_finite = all_finite and all(
                np.isfinite(value)
                for values in (
                    table["directional_mse"],
                    table["cosine"],
                    table["norm_ratio"],
                )
                for value in values
            )
            shard_manifest.append(
                {
                    "critic": critic_name,
                    "shard_index": shard_index,
                    "path": str(path),
                    "rows": len(expected_ids),
                    "sha256": sha256_file(path),
                    "cell_ids_sha256": hashlib.sha256(
                        "\n".join(expected_ids).encode("utf-8")
                    ).hexdigest(),
                }
            )
        predicted_cell_ids[critic_name] = critic_cell_ids
        summaries[critic_name] = {
            "cells": len(critic_cell_ids),
            "shards": len(shards),
            "mean_directional_mse": float(np.mean(directional_mse)),
            "mean_cosine": float(np.mean(cosine)),
            "identity_directional_mse": float(np.mean(identity_directional_mse)),
        }

    all_cell_ids = [str(cell["cell_id"]) for cell in cells]
    independent_cell_ids = [
        str(cell["cell_id"]) for cell in cells if cell["family"] in court_families
    ]
    passed = (
        len(expected_row_ids) == int(config["selection"]["deep_dive_rows"])
        and predicted_cell_ids["primary"] == all_cell_ids
        and predicted_cell_ids["independent"] == independent_cell_ids
        and all_finite
        and all(
            summary["identity_directional_mse"]
            <= float(gates["canary_ar_max_directional_mse"])
            for summary in summaries.values()
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "p2_lattice_full",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len(expected_row_ids),
        "resolved_cells": len(cells),
        "alternate_telling_cells": sum(
            cell["family"] == "alternate_telling" for cell in cells
        ),
        "summaries": summaries,
        "shards": shard_manifest,
    }
    write_json(report_path, report)
    return report


def run_alternate_tellings(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np
    import torch

    from eval_nano_av_ar_roundtrip_gate import parse_generated_explanation
    from nano_av_warmstart_smoke import load_av_config, resolve_injection_scale

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model_outputs = resolve_path(
        paths["model_outputs_dir"], config_path=config_path
    )
    output_dir = model_outputs / "p1_alternate_tellings"
    report_path = output_dir / "alternate_tellings_report.json"
    records_path = output_dir / "alternate_tellings.jsonl"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    canary_report_path = model_outputs / "e1_canary" / "canary_av_report.json"
    if not canary_report_path.is_file() or not json.loads(
        canary_report_path.read_text()
    ).get("passed"):
        raise ObservatoryConfigError("E1 AV canary must pass before sampled tellings")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        records_path.unlink(missing_ok=True)
    completed = {
        str(record["cell_id"]): record
        for record in (read_jsonl(records_path) if records_path.is_file() else [])
    }
    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    pending = [
        cell
        for cell in read_jsonl(corpus_dir / "interventions.jsonl")
        if cell["family"] == "alternate_telling"
    ]
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    sources = {int(row["row_index"]): row for row in source_rows}
    checkpoint = resolve_path(models["av_hf_staging"], config_path=config_path)
    model, tokenizer = load_av_model(
        checkpoint,
        torch_dtype=str(evaluation["torch_dtype"]),
        device_map=str(evaluation.get("av_device_map", "auto")),
    )
    av_cfg = load_av_config(
        resolve_path(paths["validation_parquet"], config_path=config_path), tokenizer
    )
    injection_scale = resolve_injection_scale(
        evaluation["injection_scale"], av_cfg.d_model
    )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for cell in pending:
        grouped.setdefault(int(cell["row_index"]), []).append(cell)
    for ordinal, (row_index, cells) in enumerate(sorted(grouped.items()), start=1):
        cells.sort(key=lambda cell: int(cell["spec"]["sample_index"]))
        missing = [cell for cell in cells if cell["cell_id"] not in completed]
        if not missing:
            continue
        if len(missing) != len(cells):
            raise ObservatoryConfigError(
                f"partial sampled-telling row cannot be resumed safely: {row_index}"
            )
        source = sources.get(row_index)
        if source is None:
            raise ObservatoryConfigError(f"missing source for sampled row {row_index}")
        print(
            f"[observatory:p1] sampled tellings row {ordinal}/{len(grouped)} "
            f"row_index={row_index}",
            flush=True,
        )
        generated = sample_generate_batch_full_prefix(
            model,
            tokenizer,
            av_cfg,
            source,
            torch.tensor(source["activation_vector"], dtype=torch.float32),
            seeds=[int(cell["spec"]["seed"]) for cell in cells],
            injection_scale=injection_scale,
            max_new_tokens=int(evaluation["max_new_tokens"]),
            temperature=float(evaluation["sampling_temperature"]),
            top_p=float(evaluation["sampling_top_p"]),
            stop_text="</explanation>",
        )
        with records_path.open("a") as handle:
            for cell, sample in zip(cells, generated, strict=True):
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "cell_id": str(cell["cell_id"]),
                    "row_id": str(cell["row_id"]),
                    "row_index": row_index,
                    "variant": str(cell["variant"]),
                    "generated": sample["text"],
                    "parsed": parse_generated_explanation(
                        sample["text"], fallback="empty"
                    ),
                    "token_ids": sample["token_ids"],
                    "token_logprobs": sample["token_logprobs"],
                    "steps": sample["steps"],
                    "seed": sample["seed"],
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                completed[record["cell_id"]] = record
    del model, tokenizer
    release_cuda_memory()
    ordered = [completed[str(cell["cell_id"])] for cell in pending]
    usable_fraction = float(np.mean([record["parsed"]["usable"] for record in ordered]))
    closed_fraction = float(np.mean([record["parsed"]["closed"] for record in ordered]))
    passed = (
        len(ordered) == len(pending) == 400
        and len({record["cell_id"] for record in ordered}) == 400
        and usable_fraction >= float(gates["alternate_parse_usable_fraction"])
        and all(np.isfinite(value) for record in ordered for value in record["token_logprobs"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "p1_alternate_tellings",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len(grouped),
        "samples": len(ordered),
        "usable_fraction": usable_fraction,
        "closed_fraction": closed_fraction,
        "sampling": {
            "backend": "full_prefix_batch",
            "temperature": float(evaluation["sampling_temperature"]),
            "top_p": float(evaluation["sampling_top_p"]),
            "max_new_tokens": int(evaluation["max_new_tokens"]),
        },
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
    }
    write_json(report_path, report)
    return report


def _behavior_evidence(
    *,
    model: Any,
    boundary_module: Any,
    tokenizer: Any,
    entry: dict[str, Any],
    baseline_continuation: list[int],
    baseline_logits: Any,
    wake_positions: int,
    continuation_tokens: int,
    generation_backend: str,
) -> dict[str, Any]:
    if generation_backend == "full_prefix":
        patched = greedy_generate_patched_full_prefix(
            model=model,
            boundary_module=boundary_module,
            prefix=entry["prefix"],
            replacement=entry["replacement"],
            max_new_tokens=continuation_tokens,
            eos_token_id=None,
        )
    elif generation_backend == "explicit_cache":
        patched = greedy_generate_patched_cached(
            model=model,
            boundary_module=boundary_module,
            tokenizer=tokenizer,
            prefix=entry["prefix"],
            replacement=entry["replacement"],
            max_new_tokens=continuation_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=None,
        )
    else:
        raise ObservatoryConfigError(
            f"unsupported functional generation backend: {generation_backend}"
        )
    wake = functional_wake_metrics(
        model=model,
        boundary_module=boundary_module,
        prefix=entry["prefix"],
        baseline_continuation=baseline_continuation,
        replacement=entry["replacement"],
        wake_positions=wake_positions,
        baseline_position_logits=baseline_logits,
    )
    return {
        "baseline_continuation_token_ids": baseline_continuation,
        "baseline_continuation_text": tokenizer.decode(
            baseline_continuation, skip_special_tokens=False
        ),
        "patched_continuation_token_ids": patched,
        "patched_continuation_text": tokenizer.decode(
            patched, skip_special_tokens=False
        ),
        "wake": wake,
        "generation_protocol": {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": continuation_tokens,
            "eos_stopping": False,
            "boundary_replacement": (
                "each_full_prefix_forward"
                if generation_backend == "full_prefix"
                else "prefill_once"
            ),
            "backend": generation_backend,
        },
    }


def run_functional_pilot(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from types import SimpleNamespace
    from transformers import PreTrainedTokenizerFast

    from nano_introspection import resolve_nano_module_paths
    from nano_r33_functional_core import rescale_direction
    from nano_r33_functional_runtime import load_target_model, run_identity_pass
    from nano_r33_source_rows import provenance_key

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model_outputs = resolve_path(
        paths["model_outputs_dir"], config_path=config_path
    )
    output_dir = model_outputs / "e4_functional_pilot"
    report_path = output_dir / "functional_pilot_report.json"
    records_path = output_dir / "functional_pilot_rows.jsonl"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    lattice_report_path = model_outputs / "e3_lattice_pilot" / "lattice_pilot_report.json"
    if not lattice_report_path.is_file() or not json.loads(
        lattice_report_path.read_text()
    ).get("passed"):
        raise ObservatoryConfigError("E3 lattice pilot must pass before E4")
    base_hf = resolve_path(models["base_hf"], config_path=config_path)
    if not hf_checkpoint_complete(base_hf):
        raise ObservatoryConfigError(f"complete base HF checkpoint is required: {base_hf}")
    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    selection = json.loads((corpus_dir / "selection_manifest.json").read_text())
    behavior_ids = set(selection["behavior_row_ids"])
    canary_ids = set(selection["canary_row_ids"])
    pilot_ids = behavior_ids & canary_ids
    if len(pilot_ids) != 4:
        raise ObservatoryConfigError(
            f"functional pilot requires four behavior canaries; found {len(pilot_ids)}"
        )
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    sources = [
        row for row in source_rows if f"validation-{int(row['row_index'])}" in pilot_ids
    ]
    sources.sort(key=lambda row: int(row["row_index"]))
    selected = [
        {
            "split": "validation",
            "row_index": int(source["row_index"]),
            "doc_id": str(source["doc_id"]),
            "token_position": int(source["token_position"]),
            "n_raw_tokens": int(source["n_raw_tokens"]),
            "content_family_id": str(source["content_family_id"]),
        }
        for source in sources
    ]
    source_by_index = {int(row["row_index"]): row for row in sources}
    cells = {
        str(cell["cell_id"]): cell
        for cell in read_jsonl(corpus_dir / "interventions.jsonl")
        if cell["row_id"] in pilot_ids
    }
    pilot_groups = {
        row_id: sorted(
            {
                str(cell["control_group_id"])
                for cell in cells.values()
                if cell["row_id"] == row_id
                and cell["family"] == "clause_swap"
                and cell.get("control_group_id")
            }
        )[0]
        for row_id in pilot_ids
    }
    lattice_predictions = pq.read_table(
        model_outputs / "e3_lattice_pilot" / "lattice_pilot_predictions.parquet"
    ).to_pylist()
    chosen = []
    for prediction in lattice_predictions:
        if prediction["critic"] != "primary" or prediction["row_id"] not in pilot_ids:
            continue
        cell = cells[str(prediction["cell_id"])]
        if cell["family"] == "identity" or (
            cell["family"] == "clause_swap"
            and str(cell.get("control_group_id")) == pilot_groups[str(cell["row_id"])]
        ):
            chosen.append((cell, prediction))
    expected_per_row = 7
    if len(chosen) != len(pilot_ids) * expected_per_row:
        raise ObservatoryConfigError(
            f"functional pilot expected {len(pilot_ids) * expected_per_row} cells, found {len(chosen)}"
        )
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    boundary = int(evaluation["boundary"])
    if layers is None or not 1 <= boundary <= len(layers):
        raise ObservatoryConfigError(f"invalid functional boundary: {boundary}")
    boundary_module = layers[boundary - 1]
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", 0)
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    identity_rows, original_logits = run_identity_pass(
        model=model,
        boundary_module=boundary_module,
        selected=selected,
        sources=sources,
        batch_size=int(evaluation["functional_batch_size"]),
        pad_token_id=int(pad_token_id or 0),
    )
    entries: list[dict[str, Any]] = []
    for cell, prediction in chosen:
        source = source_by_index[int(prediction["row_index"])]
        gold = torch.tensor(source["activation_vector"], dtype=torch.float32).view(1, -1)
        predicted = torch.tensor(
            prediction["prediction_vector"], dtype=torch.float32
        ).view(1, -1)
        replacement = rescale_direction(predicted, gold)[0]
        entries.append(
            {
                "split": "validation",
                "row_index": int(source["row_index"]),
                "provenance_key": list(provenance_key(source)),
                "content_family_id": str(source["content_family_id"]),
                "variant": str(cell["variant"]),
                "cell_id": str(cell["cell_id"]),
                "prefix": [int(token) for token in source["token_ids_prefix"]],
                "replacement": replacement,
            }
        )
    functional_rows = run_functional_pass_detailed(
        model=model,
        boundary_module=boundary_module,
        entries=entries,
        original_logits=original_logits,
        batch_size=int(evaluation["functional_batch_size"]),
        pad_token_id=int(pad_token_id or 0),
        top_k=max(int(value) for value in evaluation["top_k"]),
    )
    continuation_tokens = int(config["grid"]["continuation_tokens"])
    wake_positions = int(config["grid"]["wake_positions"])
    generation_backend = str(evaluation["functional_generation_backend"])
    baseline_continuations: dict[int, list[int]] = {}
    baseline_logits: dict[int, Any] = {}
    for source in sources:
        row_index = int(source["row_index"])
        prefix = [int(token) for token in source["token_ids_prefix"]]
        continuation = greedy_generate_unpatched(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            max_new_tokens=continuation_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=None,
            backend=generation_backend,
        )
        baseline_continuations[row_index] = continuation
        baseline_logits[row_index] = baseline_wake_logits(
            model=model,
            prefix=prefix,
            baseline_continuation=continuation,
            wake_positions=wake_positions,
        )
    entry_by_cell = {str(entry["cell_id"]): entry for entry in entries}
    for record in functional_rows:
        entry = entry_by_cell[str(record["cell_id"])]
        row_index = int(entry["row_index"])
        record.update(
            _behavior_evidence(
                model=model,
                boundary_module=boundary_module,
                tokenizer=tokenizer,
                entry=entry,
                baseline_continuation=baseline_continuations[row_index],
                baseline_logits=baseline_logits[row_index],
                wake_positions=wake_positions,
                continuation_tokens=continuation_tokens,
                generation_backend=generation_backend,
            )
        )
    generation_equivalence: list[dict[str, Any]] = []
    for entry in [entry for entry in entries if entry["variant"] == "teacher"]:
        reference = greedy_generate_patched_full_prefix(
            model=model,
            boundary_module=boundary_module,
            prefix=entry["prefix"],
            replacement=entry["replacement"],
            max_new_tokens=min(4, continuation_tokens),
            eos_token_id=None,
        )
        cached = next(
            row["patched_continuation_token_ids"]
            for row in functional_rows
            if row["cell_id"] == entry["cell_id"]
        )[: len(reference)]
        generation_equivalence.append(
            {
                "cell_id": str(entry["cell_id"]),
                "reference_token_ids": reference,
                "cached_token_ids": cached,
                "exact_match": cached == reference,
            }
        )
    del model, tokenizer
    release_cuda_memory()
    output_dir.mkdir(parents=True, exist_ok=True)
    with records_path.open("w") as handle:
        for record in functional_rows:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tolerances = {
        "relative_l2": float(gates["identity_relative_l2"]),
        "max_abs": float(gates["identity_max_abs"]),
        "one_minus_cos": float(gates["identity_one_minus_cos"]),
    }
    identity_failures = [
        row
        for row in identity_rows
        if any(
            float(row["logit_identity"][metric]) > threshold
            for metric, threshold in tolerances.items()
        )
    ]
    finite = all(
        np.isfinite(float(value))
        for row in functional_rows
        for value in row["metrics"].values()
        if isinstance(value, (int, float))
    )
    wake_finite = all(
        np.isfinite(float(value))
        for row in functional_rows
        for wake in row["wake"]
        for value in wake.values()
        if isinstance(value, (int, float))
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "e4_functional_pilot",
        "passed": (
            not identity_failures
            and finite
            and wake_finite
            and len(functional_rows) == len(entries)
            and len(generation_equivalence) == len(pilot_ids)
            and all(row["exact_match"] for row in generation_equivalence)
            and all(len(row["wake"]) == wake_positions for row in functional_rows)
            and all(
                len(row["patched_continuation_token_ids"]) == continuation_tokens
                for row in functional_rows
            )
        ),
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len(pilot_ids),
        "cells": len(entries),
        "boundary": boundary,
        "identity_tolerances": tolerances,
        "identity_rows": identity_rows,
        "identity_failures": len(identity_failures),
        "generation_equivalence": generation_equivalence,
        "wake_positions": wake_positions,
        "continuation_tokens": continuation_tokens,
        "functional_rows": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
        },
    }
    write_json(report_path, report)
    return report


def run_functional_full(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    """Run next-token, wake, and continuation evaluation for BEHAVIOR cells."""

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from types import SimpleNamespace
    from transformers import PreTrainedTokenizerFast

    from nano_introspection import resolve_nano_module_paths
    from nano_r33_functional_core import rescale_direction
    from nano_r33_functional_runtime import load_target_model, run_identity_pass
    from nano_r33_source_rows import provenance_key

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    grid = config["grid"]
    model_outputs = resolve_path(paths["model_outputs_dir"], config_path=config_path)
    output_dir = model_outputs / "p3_functional_full"
    report_path = output_dir / "functional_full_report.json"
    records_path = output_dir / "functional_full_rows.jsonl"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    for dependency, label in (
        (
            model_outputs / "e4_functional_pilot" / "functional_pilot_report.json",
            "functional pilot",
        ),
        (
            model_outputs / "p2_lattice_full" / "lattice_full_report.json",
            "full lattice",
        ),
    ):
        if not dependency.is_file() or not json.loads(dependency.read_text()).get(
            "passed"
        ):
            raise ObservatoryConfigError(f"{label} must pass before full behavior eval")
    base_hf = resolve_path(models["base_hf"], config_path=config_path)
    if not hf_checkpoint_complete(base_hf):
        raise ObservatoryConfigError(f"complete base HF checkpoint is required: {base_hf}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        records_path.unlink(missing_ok=True)
    existing_records = read_jsonl(records_path) if records_path.is_file() else []
    existing_records = [
        record
        for record in existing_records
        if isinstance(record.get("wake"), list)
        and isinstance(record.get("patched_continuation_token_ids"), list)
    ]
    if records_path.is_file():
        with records_path.open("w") as handle:
            for record in existing_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    completed = {str(record["cell_id"]): record for record in existing_records}
    if len(completed) != len(existing_records):
        raise ObservatoryConfigError("functional output contains duplicate cell ids")

    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    selection = json.loads((corpus_dir / "selection_manifest.json").read_text())
    behavior_ids = set(selection["behavior_row_ids"])
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    sources = [
        row
        for row in source_rows
        if f"validation-{int(row['row_index'])}" in behavior_ids
    ]
    sources.sort(key=lambda row: int(row["row_index"]))
    if len(sources) != int(config["selection"]["behavior_rows"]):
        raise ObservatoryConfigError(
            f"functional full expected {config['selection']['behavior_rows']} rows, "
            f"found {len(sources)}"
        )
    selected = [
        {
            "split": "validation",
            "row_index": int(source["row_index"]),
            "doc_id": str(source["doc_id"]),
            "token_position": int(source["token_position"]),
            "n_raw_tokens": int(source["n_raw_tokens"]),
            "content_family_id": str(source["content_family_id"]),
        }
        for source in sources
    ]
    source_by_index = {int(row["row_index"]): row for row in sources}
    cells = {
        str(cell["cell_id"]): cell
        for cell in read_jsonl(corpus_dir / "interventions.jsonl")
        if cell["row_id"] in behavior_ids
        and cell["depth"] == "BEHAVIOR"
        and cell["family"] in {"identity", "clause_swap"}
    }
    lattice_report = json.loads(
        (
            model_outputs / "p2_lattice_full" / "lattice_full_report.json"
        ).read_text()
    )
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for shard in lattice_report["shards"]:
        if shard["critic"] != "primary":
            continue
        for prediction in pq.read_table(shard["path"]).to_pylist():
            cell = cells.get(str(prediction["cell_id"]))
            if cell is not None:
                chosen.append((cell, prediction))
    chosen.sort(key=lambda item: (int(item[1]["row_index"]), str(item[0]["cell_id"])))
    expected_per_row = 1 + (
        int(grid["clause_chips"])
        * len(grid["clause_lanes"])
        * len(grid["behavior_doses"])
    )
    expected_cells = len(sources) * expected_per_row
    if len(chosen) != expected_cells:
        raise ObservatoryConfigError(
            f"functional full expected {expected_cells} cells, found {len(chosen)}"
        )

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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    boundary = int(evaluation["boundary"])
    if layers is None or not 1 <= boundary <= len(layers):
        raise ObservatoryConfigError(f"invalid functional boundary: {boundary}")
    boundary_module = layers[boundary - 1]
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", 0)
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    identity_rows, original_logits = run_identity_pass(
        model=model,
        boundary_module=boundary_module,
        selected=selected,
        sources=sources,
        batch_size=int(evaluation["functional_batch_size"]),
        pad_token_id=int(pad_token_id or 0),
    )
    continuation_tokens = int(grid["continuation_tokens"])
    wake_positions = int(grid["wake_positions"])
    generation_backend = str(evaluation["functional_generation_backend"])
    baseline_continuations: dict[int, list[int]] = {}
    baseline_logits: dict[int, Any] = {}
    for source in sources:
        row_index = int(source["row_index"])
        prefix = [int(token) for token in source["token_ids_prefix"]]
        continuation = greedy_generate_unpatched(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            max_new_tokens=continuation_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=None,
            backend=generation_backend,
        )
        baseline_continuations[row_index] = continuation
        baseline_logits[row_index] = baseline_wake_logits(
            model=model,
            prefix=prefix,
            baseline_continuation=continuation,
            wake_positions=wake_positions,
        )
    entries: list[dict[str, Any]] = []
    for cell, prediction in chosen:
        if str(cell["cell_id"]) in completed:
            continue
        source = source_by_index[int(prediction["row_index"])]
        gold = torch.tensor(source["activation_vector"], dtype=torch.float32).view(1, -1)
        predicted = torch.tensor(
            prediction["prediction_vector"], dtype=torch.float32
        ).view(1, -1)
        entries.append(
            {
                "split": "validation",
                "row_index": int(source["row_index"]),
                "provenance_key": list(provenance_key(source)),
                "content_family_id": str(source["content_family_id"]),
                "variant": str(cell["variant"]),
                "cell_id": str(cell["cell_id"]),
                "prefix": [int(token) for token in source["token_ids_prefix"]],
                "replacement": rescale_direction(predicted, gold)[0],
            }
        )
    functional_shard_cells = int(evaluation.get("functional_shard_cells", 32))
    try:
        for start in range(0, len(entries), functional_shard_cells):
            chunk = entries[start : start + functional_shard_cells]
            print(
                f"[observatory:p3] behavior cells "
                f"{start + 1}-{start + len(chunk)}/{len(entries)}",
                flush=True,
            )
            output_rows = run_functional_pass_detailed(
                model=model,
                boundary_module=boundary_module,
                entries=chunk,
                original_logits=original_logits,
                batch_size=int(evaluation["functional_batch_size"]),
                pad_token_id=int(pad_token_id or 0),
                top_k=max(int(value) for value in evaluation["top_k"]),
            )
            entry_by_cell = {str(entry["cell_id"]): entry for entry in chunk}
            for record in output_rows:
                entry = entry_by_cell[str(record["cell_id"])]
                row_index = int(entry["row_index"])
                record.update(
                    _behavior_evidence(
                        model=model,
                        boundary_module=boundary_module,
                        tokenizer=tokenizer,
                        entry=entry,
                        baseline_continuation=baseline_continuations[row_index],
                        baseline_logits=baseline_logits[row_index],
                        wake_positions=wake_positions,
                        continuation_tokens=continuation_tokens,
                        generation_backend=generation_backend,
                    )
                )
            with records_path.open("a") as handle:
                for record in output_rows:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    completed[str(record["cell_id"])] = record
    finally:
        del model, tokenizer
        release_cuda_memory()

    ordered = [completed[str(cell["cell_id"])] for cell, _ in chosen]
    tolerances = {
        "relative_l2": float(gates["identity_relative_l2"]),
        "max_abs": float(gates["identity_max_abs"]),
        "one_minus_cos": float(gates["identity_one_minus_cos"]),
    }
    identity_failures = [
        row
        for row in identity_rows
        if any(
            float(row["logit_identity"][metric]) > threshold
            for metric, threshold in tolerances.items()
        )
    ]
    finite = all(
        np.isfinite(float(value))
        for row in ordered
        for value in row["metrics"].values()
        if isinstance(value, (int, float))
    )
    wake_finite = all(
        np.isfinite(float(value))
        for row in ordered
        for wake in row["wake"]
        for value in wake.values()
        if isinstance(value, (int, float))
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "p3_functional_full",
        "passed": (
            not identity_failures
            and finite
            and wake_finite
            and len(ordered) == expected_cells
            and len(completed) == expected_cells
            and all(len(row["wake"]) == wake_positions for row in ordered)
            and all(
                len(row["patched_continuation_token_ids"]) == continuation_tokens
                for row in ordered
            )
        ),
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len(sources),
        "cells": len(ordered),
        "expected_cells_per_row": expected_per_row,
        "boundary": boundary,
        "identity_tolerances": tolerances,
        "identity_rows": identity_rows,
        "identity_failures": len(identity_failures),
        "wake_positions": wake_positions,
        "continuation_tokens": continuation_tokens,
        "functional_rows": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
        },
    }
    write_json(report_path, report)
    return report


def run_trace_extract(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    import numpy as np
    import torch
    from types import SimpleNamespace
    from transformers import PreTrainedTokenizerFast

    from nano_extraction_identity import tensor_metrics
    from nano_introspection import resolve_nano_module_paths
    from nano_r33_functional_runtime import load_target_model

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    model_outputs = resolve_path(
        paths["model_outputs_dir"], config_path=config_path
    )
    output_dir = model_outputs / "e5_trace_pilot"
    report_path = output_dir / "trace_extract_report.json"
    trajectories_path = output_dir / "fresh_trajectories.parquet"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    base_hf = resolve_path(models["base_hf"], config_path=config_path)
    if not hf_checkpoint_complete(base_hf):
        raise ObservatoryConfigError(f"complete base HF checkpoint is required: {base_hf}")
    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    selection = json.loads((corpus_dir / "selection_manifest.json").read_text())
    film_ids = set(selection["film_row_ids"])
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    film_rows = [
        row for row in source_rows if f"validation-{int(row['row_index'])}" in film_ids
    ]
    film_rows.sort(key=lambda row: int(row["row_index"]))
    if len(film_rows) != int(config["selection"]["film_rows"]):
        raise ObservatoryConfigError(f"expected 10 film rows, found {len(film_rows)}")
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
        raise ObservatoryConfigError(f"invalid trace boundary: {boundary}")
    boundary_module = layers[boundary - 1]
    start_device = model.get_input_embeddings().weight.device
    output_rows: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    for ordinal, source in enumerate(film_rows, start=1):
        print(
            f"[observatory:e5] fresh trace {ordinal}/{len(film_rows)} "
            f"row_index={source['row_index']}",
            flush=True,
        )
        token_ids = [int(value) for value in source["token_ids_prefix"]]
        inputs = torch.tensor([token_ids], dtype=torch.long, device=start_device)
        attention_mask = torch.ones_like(inputs)
        captured: dict[str, Any] = {}

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach().float().cpu()

        handle = boundary_module.register_forward_hook(capture)
        try:
            with torch.no_grad():
                model(input_ids=inputs, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None or tuple(hidden.shape) != (1, len(token_ids), 2688):
            raise ObservatoryConfigError(
                f"unexpected trace shape for row {source['row_index']}: "
                f"{None if hidden is None else tuple(hidden.shape)}"
            )
        positions = select_trajectory_positions(
            len(token_ids),
            minimum_context=int(evaluation["film_min_context_tokens"]),
            count=int(evaluation["film_positions_per_doc"]),
        )
        row_id = f"validation-{int(source['row_index'])}"
        for position in positions:
            output_rows.append(
                {
                    "row_id": row_id,
                    "row_index": int(source["row_index"]),
                    "doc_id": str(source["doc_id"]),
                    "content_family_id": str(source["content_family_id"]),
                    "position": position,
                    "n_context_tokens": position + 1,
                    "token_id": token_ids[position],
                    "token_text": tokenizer.decode([token_ids[position]]),
                    "activation_vector": hidden[0, position].to(torch.float16).tolist(),
                }
            )
        stored = torch.tensor(source["activation_vector"], dtype=torch.float32)
        drift.append(
            {
                "row_id": row_id,
                "row_index": int(source["row_index"]),
                **tensor_metrics(hidden[0, -1], stored),
            }
        )
    del model, tokenizer
    release_cuda_memory()
    write_trajectory_parquet(trajectories_path, output_rows)
    passed = (
        len({row["row_id"] for row in output_rows}) == 10
        and all(np.isfinite(value) for row in output_rows for value in row["activation_vector"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "e5_trace_extract",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "claim_scope": "fresh_forward_exploratory",
        "documents": len(film_rows),
        "trajectory_rows": len(output_rows),
        "boundary": boundary,
        "stored_final_position_drift": drift,
        "trajectories": {
            "path": str(trajectories_path),
            "sha256": sha256_file(trajectories_path),
        },
    }
    write_json(report_path, report)
    return report


def run_trace_describe(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    """Generate AV descriptions for every precomputed film-set position."""

    import numpy as np
    import pyarrow.parquet as pq
    import torch

    from eval_nano_av_ar_roundtrip_gate import parse_generated_explanation
    from nano_av_warmstart_smoke import (
        generate_controls_for_row,
        load_av_config,
        resolve_injection_scale,
    )

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model_outputs = resolve_path(paths["model_outputs_dir"], config_path=config_path)
    output_dir = model_outputs / "p1_trace_descriptions"
    report_path = output_dir / "trace_descriptions_report.json"
    records_path = output_dir / "trace_descriptions.jsonl"
    if reusable := _reusable_report(report_path, config, force=force):
        return reusable
    trace_report_path = model_outputs / "e5_trace_pilot" / "trace_extract_report.json"
    av_report_path = model_outputs / "e1_canary" / "canary_av_report.json"
    for dependency, label in (
        (trace_report_path, "trace extraction"),
        (av_report_path, "AV canary"),
    ):
        if not dependency.is_file() or not json.loads(dependency.read_text()).get(
            "passed"
        ):
            raise ObservatoryConfigError(f"{label} must pass before trace descriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        records_path.unlink(missing_ok=True)
    existing_records = read_jsonl(records_path) if records_path.is_file() else []
    completed = {
        (int(record["row_index"]), int(record["position"])): record
        for record in existing_records
    }
    if len(completed) != len(existing_records):
        raise ObservatoryConfigError("trace descriptions contain duplicate position keys")

    trajectories_path = model_outputs / "e5_trace_pilot" / "fresh_trajectories.parquet"
    trajectories = pq.read_table(trajectories_path).to_pylist()
    source_rows = read_parquet_rows(
        resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    )
    sources = {int(row["row_index"]): row for row in source_rows}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        grouped.setdefault(int(trajectory["row_index"]), []).append(trajectory)
    checkpoint = resolve_path(models["av_hf_staging"], config_path=config_path)
    model, tokenizer = load_av_model(
        checkpoint,
        torch_dtype=str(evaluation["torch_dtype"]),
        device_map=str(evaluation.get("av_device_map", "auto")),
    )
    av_cfg = load_av_config(
        resolve_path(paths["validation_parquet"], config_path=config_path), tokenizer
    )
    injection_scale = resolve_injection_scale(
        evaluation["injection_scale"], av_cfg.d_model
    )
    generation_batch_size = int(evaluation.get("trace_generation_batch_size", 5))
    if generation_batch_size <= 0:
        raise ObservatoryConfigError("trace_generation_batch_size must be positive")
    try:
        for row_ordinal, (row_index, row_trajectories) in enumerate(
            sorted(grouped.items()), start=1
        ):
            source = sources.get(row_index)
            if source is None:
                raise ObservatoryConfigError(
                    f"trace trajectory has no selected source row: {row_index}"
                )
            row_trajectories.sort(key=lambda row: int(row["position"]))
            pending = [
                row
                for row in row_trajectories
                if (row_index, int(row["position"])) not in completed
            ]
            print(
                f"[observatory:p1] trace descriptions row "
                f"{row_ordinal}/{len(grouped)} pending={len(pending)}",
                flush=True,
            )
            for start in range(0, len(pending), generation_batch_size):
                chunk = pending[start : start + generation_batch_size]
                names = [f"position_{int(row['position']):05d}" for row in chunk]
                controls = {
                    name: torch.tensor(row["activation_vector"], dtype=torch.float32)
                    for name, row in zip(names, chunk, strict=True)
                }
                generated = generate_controls_for_row(
                    model,
                    tokenizer,
                    av_cfg,
                    source,
                    controls,
                    names,
                    injection_scale=injection_scale,
                    max_new_tokens=int(evaluation["max_new_tokens"]),
                    generation_prefix="",
                    stop_text="</explanation>",
                    use_cache=False,
                    batch_full_prefix=True,
                )
                with records_path.open("a") as handle:
                    for name, trajectory in zip(names, chunk, strict=True):
                        record = {
                            "schema_version": SCHEMA_VERSION,
                            "row_id": str(trajectory["row_id"]),
                            "row_index": row_index,
                            "doc_id": str(trajectory["doc_id"]),
                            "content_family_id": str(
                                trajectory["content_family_id"]
                            ),
                            "position": int(trajectory["position"]),
                            "n_context_tokens": int(
                                trajectory["n_context_tokens"]
                            ),
                            "generated": generated[name],
                            "parsed": parse_generated_explanation(
                                generated[name], fallback="empty"
                            ),
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()
                        completed[(row_index, int(trajectory["position"]))] = record
    finally:
        del model, tokenizer
        release_cuda_memory()

    ordered = [
        completed[(int(row["row_index"]), int(row["position"]))]
        for row in trajectories
    ]
    usable_fraction = float(np.mean([row["parsed"]["usable"] for row in ordered]))
    closed_fraction = float(np.mean([row["parsed"]["closed"] for row in ordered]))
    passed = (
        len(ordered) == len(trajectories)
        and len(completed) == len(trajectories)
        and usable_fraction >= float(gates["alternate_parse_usable_fraction"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "p1_trace_descriptions",
        "passed": passed,
        "config_sha256": config_fingerprint(config),
        "split": "validation",
        "rows": len(grouped),
        "positions": len(ordered),
        "usable_fraction": usable_fraction,
        "closed_fraction": closed_fraction,
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
    }
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "canary-av",
            "canary-ar",
            "token-logprobs",
            "lattice-pilot",
            "lattice-full",
            "alternate-tellings",
            "functional-pilot",
            "functional-full",
            "trace-extract",
            "trace-describe",
            "poetry-prepare",
            "poetry-extract",
            "poetry-describe",
            "poetry-score",
            "poetry-reconstruct",
            "poetry-intervene",
        ),
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    _configure_runtime()
    try:
        if args.phase == "canary-av":
            report = run_canary_av(args.config, force=args.force)
        elif args.phase == "canary-ar":
            report = run_canary_ar(args.config, force=args.force)
        elif args.phase == "lattice-pilot":
            report = run_lattice_pilot(args.config, force=args.force)
        elif args.phase == "lattice-full":
            report = run_lattice_full(args.config, force=args.force)
        elif args.phase == "alternate-tellings":
            report = run_alternate_tellings(args.config, force=args.force)
        elif args.phase == "functional-pilot":
            report = run_functional_pilot(args.config, force=args.force)
        elif args.phase == "functional-full":
            report = run_functional_full(args.config, force=args.force)
        elif args.phase == "trace-extract":
            report = run_trace_extract(args.config, force=args.force)
        elif args.phase == "trace-describe":
            report = run_trace_describe(args.config, force=args.force)
        elif args.phase.startswith("poetry-"):
            from .poetry_planning import PHASE_RUNNERS

            report = PHASE_RUNNERS[args.phase](args.config, force=args.force)
        else:
            report = run_token_logprobs(args.config, force=args.force)
    except (OSError, ValueError, ObservatoryConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
