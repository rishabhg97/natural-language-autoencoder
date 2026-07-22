#!/usr/bin/env python3
"""Generate and score an oracle best-of-N Nano AV round-trip diagnostic."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import eval_nano_av_ar_roundtrip_gate as roundtrip  # noqa: E402
import nano_roundtrip_eval_config  # noqa: E402
import nano_roundtrip_queue  # noqa: E402


SCHEMA_VERSION = "nano_roundtrip_best_of_n.v1"


class BestOfNError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise BestOfNError(f"config must use schema_version {SCHEMA_VERSION}")
    for section in ("paths", "sampling", "evaluation", "execution"):
        if not isinstance(config.get(section), dict):
            raise BestOfNError(f"config requires mapping: {section}")
    samples = int(config["sampling"].get("samples_per_row", 0))
    n_values = [int(value) for value in config["sampling"].get("n_values", [])]
    if samples <= 0 or not n_values or min(n_values) <= 0 or max(n_values) > samples:
        raise BestOfNError("sampling.n_values must be positive and <= samples_per_row")
    return config


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _prepare_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    code_root = Path(config["paths"]["code_root"])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{code_root / 'scripts'}:{code_root}"
    queue_doc = {"defaults": {"av_checkpoint_prepare": config["av_checkpoint_prepare"]}}
    prepared = nano_roundtrip_queue.prepare_av_checkpoint(
        queue_doc, code_root=code_root, env=env
    )
    if prepared is None:
        raise BestOfNError("av_checkpoint_prepare did not return a prepared checkpoint")
    return prepared


def _sample_protocol(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    sampling = config["sampling"]
    evaluation = config["evaluation"]
    return {
        "schema_version": "nano_roundtrip_generation_protocol.v2",
        "backend": "full_prefix_batch_sampling",
        "prefix": "",
        "prefix_sha256": hashlib.sha256(b"").hexdigest(),
        "stop_text": "</explanation>",
        "max_new_tokens": int(evaluation["max_new_tokens"]),
        "do_sample": True,
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "seed": int(sampling["seed"]),
        "injection_scale": str(evaluation.get("injection_scale", "75")),
        "torch_dtype": str(evaluation.get("torch_dtype", "bfloat16")),
        "model_fingerprint": prepared["av_model_fingerprint"],
        "tokenizer_fingerprint": prepared["av_tokenizer_fingerprint"],
    }


def _sample_provenance(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    datasets = {
        "train": roundtrip.file_provenance(paths["train_parquet"]),
        "validation": roundtrip.file_provenance(paths["validation_parquet"]),
    }
    provenance = {
        "model_fingerprint": prepared["av_model_fingerprint"],
        "checkpoint": str(prepared["output_hf_dir"]),
        "model_revision": None,
        "tokenizer_revision": None,
        "tokenizer_fingerprint": prepared["av_tokenizer_fingerprint"],
        "datasets": datasets,
    }
    provenance["dataset_bundle_sha256"] = hashlib.sha256(
        json.dumps(datasets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return provenance


def _source_records_for_selected(
    source_records: list[dict[str, Any]], selected: list[int]
) -> dict[int, dict[str, Any]]:
    source_by_index: dict[int, dict[str, Any]] = {}
    for record in source_records:
        row_index = int(record["row_index"])
        if row_index in source_by_index:
            raise BestOfNError(f"duplicate source row_index: {row_index}")
        source_by_index[row_index] = record
    missing = [row_index for row_index in selected if row_index not in source_by_index]
    if missing:
        raise BestOfNError(
            "source generated JSONL does not cover selected validation rows: "
            f"missing={missing[:10]} count={len(missing)}"
        )
    return source_by_index


def generate(config: dict[str, Any]) -> dict[str, Any]:
    generation_device = config["execution"].get("generation_gpu_device")
    if generation_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(generation_device)

    import torch
    from nano_av_warmstart_smoke import load_av_config, resolve_injection_scale
    from observatory.model_runtime import (
        load_av_model,
        release_cuda_memory,
        sample_generate_batch_full_prefix,
    )

    prepared = _prepare_checkpoint(config)
    paths = config["paths"]
    evaluation = config["evaluation"]
    sampling = config["sampling"]
    source_records = roundtrip.read_generated_jsonl(Path(paths["source_generated_jsonl"]))
    rows, _, validation_indices, test_indices = roundtrip.load_eval_rows(
        Path(paths["train_parquet"]),
        Path(paths["validation_parquet"]),
        None,
        eval_splits=["validation"],
        content_family_manifest=Path(paths["content_family_manifest"]),
        content_family_coverage=Path(paths["content_family_coverage"]),
        require_family_disjoint_splits=True,
    )
    selected = roundtrip.select_eval_indices_by_split(
        rows,
        validation_indices=validation_indices,
        test_indices=test_indices,
        validation_limit=int(evaluation["validation_limit"]),
        test_limit=1,
        eval_splits=["validation"],
        strategy="family_stratified",
        seed=int(evaluation["selection_seed"]),
    )["validation"]
    source_by_index = _source_records_for_selected(source_records, selected)
    checkpoint = Path(prepared["output_hf_dir"])
    model, tokenizer = load_av_model(
        checkpoint,
        torch_dtype=str(evaluation.get("torch_dtype", "bfloat16")),
        device_map=str(evaluation.get("av_device_map", "cuda:0")),
    )
    av_config = load_av_config(Path(paths["validation_parquet"]), tokenizer)
    injection_scale = resolve_injection_scale(
        evaluation.get("injection_scale", "75"), av_config.d_model
    )
    sample_count = int(sampling["samples_per_row"])
    output_root = Path(paths["output_root"])
    sample_records: list[list[dict[str, Any]]] = [[] for _ in range(sample_count)]
    protocol = _sample_protocol(config, prepared)
    protocol_hash = roundtrip.generation_protocol_sha256(protocol)
    provenance = _sample_provenance(config, prepared)
    provenance_hash = roundtrip.generation_provenance_sha256(provenance)
    for ordinal, row_index in enumerate(selected, start=1):
        row = rows[row_index]
        base_seed = int(sampling["seed"]) + ordinal * 1009
        seeds = [base_seed + index for index in range(sample_count)]
        samples = sample_generate_batch_full_prefix(
            model,
            tokenizer,
            av_config,
            row,
            torch.tensor(row["activation_vector"], dtype=torch.float32),
            seeds=seeds,
            injection_scale=injection_scale,
            max_new_tokens=int(evaluation["max_new_tokens"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            stop_text="</explanation>",
        )
        for sample_index, sample in enumerate(samples):
            item = copy.deepcopy(source_by_index[row_index])
            generated = str(sample["text"])
            item["controls"] = {
                "real": {
                    "generated": generated,
                    "parsed": roundtrip.parse_generated_explanation(generated, fallback="raw"),
                    "text_overlap": roundtrip.text_overlap_metrics(
                        generated, str(item["target_explanation"])
                    ),
                }
            }
            item["generation_protocol"] = protocol
            item["generation_protocol_sha256"] = protocol_hash
            item["generation_provenance"] = provenance
            item["generation_provenance_sha256"] = provenance_hash
            item["best_of_n_sample"] = {
                "sample_index": sample_index,
                "seed": int(sample["seed"]),
                "token_logprobs": sample["token_logprobs"],
                "steps": int(sample["steps"]),
            }
            sample_records[sample_index].append(item)
        print(f"[best-of-n] row {ordinal}/{len(selected)}", flush=True)
    del model, tokenizer
    release_cuda_memory()
    generated_paths = []
    for sample_index, records in enumerate(sample_records):
        path = output_root / "generated" / f"sample_{sample_index:02d}.jsonl"
        roundtrip.write_generated_jsonl(path, records)
        generated_paths.append(str(path))
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "generate",
        "rows": len(selected),
        "samples_per_row": sample_count,
        "total_generations": len(selected) * sample_count,
        "generation_protocol": protocol,
        "generation_protocol_sha256": protocol_hash,
        "generation_gpu_device": (
            str(generation_device) if generation_device is not None else None
        ),
        "generated_jsonl": generated_paths,
    }
    _write_json(output_root / "generation_protocol.json", protocol)
    _write_json(output_root / "generation_report.json", report)
    return report


def _score_config(config: dict[str, Any], sample_index: int) -> dict[str, Any]:
    paths = config["paths"]
    evaluation = config["evaluation"]
    output_root = Path(paths["output_root"])
    return {
        "schema_version": "nano_roundtrip_eval.v1",
        "python": str(config["execution"]["python"]),
        "paths": {
            "code_root": str(paths["code_root"]),
            "av_hf_checkpoint": str(config["av_checkpoint_prepare"]["output_hf_dir"]),
            "ar_checkpoint_dir": str(paths["ar_checkpoint_dir"]),
            "train_parquet": str(paths["train_parquet"]),
            "validation_parquet": str(paths["validation_parquet"]),
            "content_family_manifest": str(paths["content_family_manifest"]),
            "content_family_coverage": str(paths["content_family_coverage"]),
            "generated_jsonl": str(output_root / "generated" / f"sample_{sample_index:02d}.jsonl"),
            "expected_generation_protocol_json": str(
                output_root / "generation_protocol.json"
            ),
            "report_json": str(output_root / "scores" / f"sample_{sample_index:02d}.json"),
        },
        "eval": {
            "eval_splits": ["validation"],
            "validation_limit": int(evaluation["validation_limit"]),
            "generation_controls": ["real"],
            "max_new_tokens": int(evaluation["max_new_tokens"]),
            "seed": int(config["sampling"]["seed"]),
            "generation_prefix": "",
            "stop_text": "</explanation>",
            "generated_text_fallback": "raw",
            "generation_backend": "legacy_batch",
            "reuse_generated": True,
            "injection_scale": str(evaluation.get("injection_scale", "75")),
            "ar_batch_size": int(evaluation.get("ar_batch_size", 4)),
            "ar_max_length": int(evaluation.get("ar_max_length", 1152)),
            "torch_dtype": str(evaluation.get("torch_dtype", "bfloat16")),
            "control_margin": 0.000001,
            "selection_strategy": "family_stratified",
            "selection_seed": int(evaluation["selection_seed"]),
            "require_family_level_inference": True,
            "min_independent_families": int(evaluation["min_independent_families"]),
            "av_model_fingerprint": str(config["av_checkpoint_prepare"]["expected_model_fingerprint"]),
            "av_tokenizer_fingerprint": str(config["av_checkpoint_prepare"]["expected_tokenizer_fingerprint"]),
            "require_generation_protocol_match": False,
            "local_files_only": True,
            "ar_device_map": "cuda:0",
            "ar_low_cpu_mem_usage": True,
        },
    }


def score(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    sample_count = int(config["sampling"]["samples_per_row"])
    devices = [str(value) for value in config["execution"]["gpu_devices"]]
    max_parallel = min(int(config["execution"].get("max_parallel", len(devices))), len(devices))

    def run_one(sample_index: int) -> dict[str, Any]:
        eval_config = _score_config(config, sample_index)
        config_path = output_root / "score_configs" / f"sample_{sample_index:02d}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))
        command = nano_roundtrip_eval_config.build_command(
            nano_roundtrip_eval_config.validate_config(eval_config), config_path=config_path
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices[sample_index % len(devices)]
        env["WANDB_MODE"] = "offline"
        log_path = output_root / "scores" / f"sample_{sample_index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            result = subprocess.run(
                command,
                cwd=Path(config["paths"]["code_root"]),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {"sample_index": sample_index, "returncode": result.returncode, "log": str(log_path)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        results = list(pool.map(run_one, range(sample_count)))
    report = {"schema_version": SCHEMA_VERSION, "phase": "score", "passed": all(item["returncode"] == 0 for item in results), "items": results}
    _write_json(output_root / "score_report.json", report)
    if report["passed"] and bool(config["execution"].get("cleanup_prepared_hf_after_score", True)):
        shutil.rmtree(Path(config["av_checkpoint_prepare"]["output_hf_dir"]), ignore_errors=True)
    return report


def summarize_best_of_n(
    sample_losses: np.ndarray, *, n_values: list[int]
) -> dict[str, Any]:
    if sample_losses.ndim != 2:
        raise BestOfNError("sample_losses must have shape [samples, rows]")
    output: dict[str, Any] = {}
    for n in n_values:
        selected = np.min(sample_losses[:n], axis=0)
        output[str(n)] = {
            "oracle_directional_nmse": float(selected.mean()),
            "rowwise": selected.tolist(),
        }
    return output


def matched_report_directional_nmse(
    report: dict[str, Any], row_indices: list[int]
) -> float:
    return float(matched_report_directional_losses(report, row_indices).mean())


def matched_report_directional_losses(
    report: dict[str, Any], row_indices: list[int]
) -> np.ndarray:
    split = report["splits"]["validation"]
    report_indices = [int(value) for value in split["row_indices"]]
    losses = list(split["rowwise_directional_mse"]["av_real"])
    if len(report_indices) != len(losses) or len(set(report_indices)) != len(
        report_indices
    ):
        raise BestOfNError("baseline report has invalid rowwise identities")
    by_index = dict(zip(report_indices, losses))
    missing = [int(value) for value in row_indices if int(value) not in by_index]
    if missing:
        raise BestOfNError(
            f"baseline report does not cover sampled rows: {missing[:10]}"
        )
    return np.asarray([by_index[int(value)] for value in row_indices], dtype=np.float64)


def family_bootstrap_summary(
    differences: np.ndarray,
    content_family_ids: list[Any],
    *,
    samples: int,
    seed: int,
    positive_interpretation: str,
) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    if len(differences) != len(content_family_ids):
        raise BestOfNError("family and rowwise metric lengths differ")
    family_values: dict[str, list[float]] = {}
    for family, value in zip(content_family_ids, differences):
        family_values.setdefault(str(family), []).append(float(value))
    family_means = np.asarray(
        [np.mean(values) for _, values in sorted(family_values.items())],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(family_means), size=(samples, len(family_means)))
    means = family_means[indices].mean(axis=1)
    return {
        "positive_interpretation": positive_interpretation,
        "rows": len(differences),
        "families": len(family_means),
        "mean": float(differences.mean()),
        "family_equal_weight_mean": float(family_means.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "row_win_fraction": float(np.mean(differences > 0.0)),
    }


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    sample_count = int(config["sampling"]["samples_per_row"])
    reports = [
        json.loads((output_root / "scores" / f"sample_{index:02d}.json").read_text())
        for index in range(sample_count)
    ]
    row_indices = reports[0]["splits"]["validation"]["row_indices"]
    content_family_ids = reports[0]["splits"]["validation"][
        "content_family_ids"
    ]
    losses = []
    for report in reports:
        split = report["splits"]["validation"]
        if split["row_indices"] != row_indices:
            raise BestOfNError("sample score reports have different row identities")
        if split["content_family_ids"] != content_family_ids:
            raise BestOfNError("sample score reports have different family identities")
        losses.append(split["rowwise_directional_mse"]["av_real"])
    sample_losses = np.asarray(losses, dtype=np.float64)
    curves = summarize_best_of_n(
        sample_losses, n_values=[int(value) for value in config["sampling"]["n_values"]]
    )
    baseline_losses: dict[str, np.ndarray] = {}
    for label, path_key in (
        ("greedy_sft", "greedy_sft_report"),
        ("matched_rl", "matched_rl_report"),
    ):
        baseline_report = json.loads(Path(config["paths"][path_key]).read_text())
        baseline_losses[label] = matched_report_directional_losses(
            baseline_report, row_indices
        )
    analysis_config = config.get("analysis") or {}
    bootstrap_samples = int(analysis_config.get("bootstrap_samples", 10000))
    bootstrap_seed = int(analysis_config.get("seed", config["sampling"]["seed"]))
    for curve in curves.values():
        selected = np.asarray(curve["rowwise"], dtype=np.float64)
        curve["paired_comparisons"] = {}
        for label, values in baseline_losses.items():
            value = float(values.mean())
            curve[f"improvement_vs_{label}"] = value - curve["oracle_directional_nmse"]
            curve["paired_comparisons"][label] = family_bootstrap_summary(
                values - selected,
                content_family_ids,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                positive_interpretation=(
                    f"oracle best-of-N has lower loss than {label}"
                ),
            )
    joint_gain = float(
        baseline_losses["greedy_sft"].mean()
        - baseline_losses["matched_rl"].mean()
    )
    for curve in curves.values():
        curve["fraction_of_matched_rl_gain_explained"] = (
            curve["improvement_vs_greedy_sft"] / joint_gain
            if joint_gain > 0.0
            else None
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "analyze",
        "rows": len(row_indices),
        "samples_per_row": sample_count,
        "sampling": config["sampling"],
        "mean_sample_directional_nmse": float(sample_losses.mean()),
        "matched_baseline_directional_nmse": {
            label: float(values.mean()) for label, values in baseline_losses.items()
        },
        "matched_rl_gain_over_greedy_sft": joint_gain,
        "oracle_curves": curves,
        "interpretation_guard": "Oracle best-of-N is a diagnostic upper bound, not a deployable model score.",
    }
    _write_json(output_root / "analysis_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "score", "analyze"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    result = {"generate": generate, "score": score, "analyze": analyze}[args.command](config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
