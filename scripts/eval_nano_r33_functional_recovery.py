#!/usr/bin/env python3
"""Evaluate whether reconstructed boundary activations preserve Nano behavior."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_eval_core import (  # noqa: E402
    clustered_paired_bootstrap_improvement,
    paired_bootstrap_improvement,
)
from eval_nano_av_ar_roundtrip_gate import (  # noqa: E402
    MODEL_FINGERPRINT_RE,
    TOKENIZER_FINGERPRINT_RE,
    compare_generation_protocols,
    generation_protocol_sha256,
    generation_provenance_sha256,
    validate_generated_record_protocols,
    validate_generated_record_provenance,
)
from nano_functional_eval_data import (  # noqa: E402
    FunctionalEvaluationError,
    build_variant_entries,
    load_ar_predictions,
    read_generated_jsonl,
    select_exact_split_rows,
    source_mean_activation,
    source_schema,
)
from nano_r33_functional_runtime import (  # noqa: E402
    load_target_model,
    run_functional_pass,
    run_identity_pass,
)
from nano_r33_source_rows import provenance_key, resolve_source_rows  # noqa: E402


SCHEMA_VERSION = "nano_r33_functional_recovery.v2"
LOWER_IS_BETTER = (
    "kl_original_to_patched",
    "js_divergence",
    "original_top1_rank",
)
HIGHER_IS_BETTER = (
    "logit_pearson",
    "top_10_overlap",
    "top_50_overlap",
)
SUMMARY_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


def validate_generation_identity(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        protocol = validate_generated_record_protocols(
            records,
            expected_protocol=None,
            require=True,
        )
        expected_provenance = records[0].get("generation_provenance") if records else None
        if not isinstance(expected_provenance, dict):
            raise ValueError("missing generation provenance")
        provenance = validate_generated_record_provenance(
            records,
            expected_provenance=expected_provenance,
            require=True,
        )
    except ValueError as exc:
        raise FunctionalEvaluationError(
            f"{label} generation identity is invalid: {exc}"
        ) from exc
    if protocol is None or provenance is None:
        raise FunctionalEvaluationError(f"{label} generation identity is missing")
    if str(protocol.get("prefix") or ""):
        raise FunctionalEvaluationError(
            f"{label} generation protocol must use an empty prefix"
        )
    model_fingerprint = str(provenance.get("model_fingerprint") or "")
    tokenizer_fingerprint = str(provenance.get("tokenizer_fingerprint") or "")
    if not MODEL_FINGERPRINT_RE.fullmatch(model_fingerprint):
        raise FunctionalEvaluationError(
            f"{label} generation model fingerprint is not content-addressed"
        )
    if not TOKENIZER_FINGERPRINT_RE.fullmatch(tokenizer_fingerprint):
        raise FunctionalEvaluationError(
            f"{label} generation tokenizer fingerprint is not content-addressed"
        )
    return {
        "protocol": protocol,
        "protocol_sha256": generation_protocol_sha256(protocol),
        "provenance": provenance,
        "provenance_sha256": generation_provenance_sha256(provenance),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _metric_passes(metrics: dict[str, Any], tolerances: dict[str, float]) -> bool:
    required = ("relative_l2", "max_abs", "one_minus_cos")
    try:
        return all(
            math.isfinite(float(metrics[name]))
            and float(metrics[name]) <= float(tolerances[name])
            for name in required
        )
    except (KeyError, TypeError, ValueError):
        return False


def _provenance_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    value = row.get("provenance_key")
    if not isinstance(value, (list, tuple)) or not value:
        raise FunctionalEvaluationError("row is missing provenance_key")
    return tuple(value)


def _summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means: dict[str, float] = {}
    for metric in SUMMARY_METRICS:
        values = [float(row["metrics"][metric]) for row in rows if metric in row["metrics"]]
        if values:
            means[metric] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return {"row_count": len(rows), "means": means, "rows": rows}


def _paired_against_sft(
    sft_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    sft_by_key = {_provenance_tuple(row): row for row in sft_rows}
    candidate_by_key = {_provenance_tuple(row): row for row in candidate_rows}
    shared = sorted(sft_by_key.keys() & candidate_by_key.keys(), key=repr)
    if not shared:
        return {"row_count": 0, "metrics": {}}

    output: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        sft = np.asarray(
            [float(sft_by_key[key]["metrics"][metric]) for key in shared],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [float(candidate_by_key[key]["metrics"][metric]) for key in shared],
            dtype=np.float64,
        )
        if metric in HIGHER_IS_BETTER:
            sft = -sft
            candidate = -candidate
        cluster_ids = [
            str(candidate_by_key[key].get("content_family_id") or "")
            for key in shared
        ]
        if all(cluster_ids) and len(set(cluster_ids)) >= 2:
            output[metric] = clustered_paired_bootstrap_improvement(
                sft,
                candidate,
                cluster_ids,
                seed=seed,
                resamples=bootstrap_resamples,
            )
        else:
            output[metric] = paired_bootstrap_improvement(
                sft,
                candidate,
                seed=seed,
                resamples=bootstrap_resamples,
            )
    return {"row_count": len(shared), "metrics": output}


def build_functional_report(
    *,
    identity_rows: list[dict[str, Any]],
    functional_rows: list[dict[str, Any]],
    identity_tolerances: dict[str, float],
    metadata: dict[str, Any],
    bootstrap_resamples: int,
    seed: int = 0,
) -> dict[str, Any]:
    failures = [
        row
        for row in identity_rows
        if not _metric_passes(row.get("logit_identity", {}), identity_tolerances)
    ]
    stored_drift_outliers = [
        row
        for row in identity_rows
        if not _metric_passes(
            row.get("stored_activation_drift", row.get("activation_identity", {})),
            identity_tolerances,
        )
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "identity_tolerances": identity_tolerances,
        "identity_rows": identity_rows,
        "gate": {
            "identity_passed": not failures,
            "reinjection_identity_passed": not failures,
            "identity_row_count": len(identity_rows),
            "failing_row_count": len(failures),
            "failing_provenance_keys": [row.get("provenance_key") for row in failures],
            "stored_activation_replay_within_tolerance": not stored_drift_outliers,
            "stored_drift_outlier_count": len(stored_drift_outliers),
            "stored_drift_outlier_keys": [
                row.get("provenance_key") for row in stored_drift_outliers
            ],
        },
        "splits": {},
    }
    if failures:
        return report

    for split in sorted({str(row["split"]) for row in functional_rows}):
        split_rows = [row for row in functional_rows if str(row["split"]) == split]
        rows_by_variant = {
            variant: [row for row in split_rows if str(row["variant"]) == variant]
            for variant in sorted({str(row["variant"]) for row in split_rows})
        }
        paired: dict[str, Any] = {}
        if "sft" in rows_by_variant:
            for variant, rows in rows_by_variant.items():
                if variant == "sft":
                    continue
                comparison = _paired_against_sft(
                    rows_by_variant["sft"],
                    rows,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                )
                paired[variant] = comparison["metrics"]
        paired_vs_stored_gold: dict[str, Any] = {}
        if "stored_gold" in rows_by_variant:
            for variant, rows in rows_by_variant.items():
                if variant == "stored_gold":
                    continue
                comparison = _paired_against_sft(
                    rows_by_variant["stored_gold"],
                    rows,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                )
                paired_vs_stored_gold[variant] = comparison["metrics"]
        paired_candidate_vs_variants: dict[str, Any] = {}
        if "candidate" in rows_by_variant:
            for variant, rows in rows_by_variant.items():
                if variant == "candidate":
                    continue
                comparison = _paired_against_sft(
                    rows,
                    rows_by_variant["candidate"],
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                )
                paired_candidate_vs_variants[variant] = comparison["metrics"]
        report["splits"][split] = {
            "variants": {
                variant: _summarize_variant_rows(rows)
                for variant, rows in rows_by_variant.items()
            },
            "paired_vs_sft": paired,
            "paired_vs_stored_gold": paired_vs_stored_gold,
            "paired_candidate_vs_variants": paired_candidate_vs_variants,
        }
    return report


def _config_hash(args: argparse.Namespace) -> str:
    if args.config_path and args.config_path.is_file():
        payload = args.config_path.read_bytes()
    else:
        payload = json.dumps(vars(args), default=str, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _release_target(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from nano_introspection import resolve_nano_module_paths

    generated_records = read_generated_jsonl(args.generated_jsonl)
    generation_identity = None
    sft_generation_identity = None
    generation_protocol_parity = None
    if args.require_generation_identity:
        generation_identity = validate_generation_identity(
            generated_records,
            label="candidate",
        )
        if args.sft_generated_jsonl:
            sft_generation_identity = validate_generation_identity(
                read_generated_jsonl(args.sft_generated_jsonl),
                label="SFT baseline",
            )
            generation_protocol_parity = compare_generation_protocols(
                generation_identity["protocol"],
                sft_generation_identity["protocol"],
            )
            if not generation_protocol_parity["matched"] or not generation_protocol_parity[
                "publication_compatible"
            ]:
                raise FunctionalEvaluationError(
                    "candidate and SFT generation protocols are not publication-compatible"
                )
    selected = select_exact_split_rows(
        generated_records,
        args.validation_limit,
        args.test_limit,
        eval_splits=args.eval_splits,
        selection_strategy=args.selection_strategy,
    )
    independent_family_count = len(
        {
            str(record.get("content_family_id") or "")
            for record in selected
            if str(record.get("content_family_id") or "")
        }
    )
    if independent_family_count < args.min_independent_families:
        raise FunctionalEvaluationError(
            "functional evaluation has too few independent content families: "
            f"observed={independent_family_count} "
            f"required={args.min_independent_families}"
        )
    source_map = resolve_source_rows(
        args.source_base_parquet,
        selected,
        batch_size=args.source_batch_size,
    )
    sources = [source_map[provenance_key(record)] for record in selected]
    predictions, critic_template, ar_hf_dir = load_ar_predictions(
        args,
        selected,
        sources,
    )
    mean_activation = source_mean_activation(
        args.mean_activation_parquet or args.source_base_parquet,
        batch_size=args.source_batch_size,
    )

    model = load_target_model(args)
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    if layers is None or not 1 <= args.boundary <= len(layers):
        _release_target(model)
        raise FunctionalEvaluationError(
            f"boundary {args.boundary} is invalid for {0 if layers is None else len(layers)} layers"
        )
    boundary_module = layers[args.boundary - 1]
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    pad_token_id = int(pad_token_id or 0)

    identity_rows, original_logits = run_identity_pass(
        model=model,
        boundary_module=boundary_module,
        selected=selected,
        sources=sources,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
    )
    tolerances = {
        "relative_l2": args.identity_relative_l2,
        "max_abs": args.identity_max_abs,
        "one_minus_cos": args.identity_one_minus_cos,
    }
    metadata = {
        "generated_jsonl": str(args.generated_jsonl),
        "sft_generated_jsonl": str(args.sft_generated_jsonl)
        if args.sft_generated_jsonl
        else None,
        "ar_checkpoint_dir": str(args.ar_checkpoint_dir),
        "ar_hf_dir": ar_hf_dir,
        "source_base_parquet": str(args.source_base_parquet),
        "mean_activation_parquet": str(
            args.mean_activation_parquet or args.source_base_parquet
        ),
        "source_schema": source_schema(args.source_base_parquet),
        "target_model": args.target_model,
        "boundary": args.boundary,
        "eval_splits": list(args.eval_splits),
        "selection_strategy": args.selection_strategy,
        "independent_family_count": independent_family_count,
        "min_independent_families": args.min_independent_families,
        "generation_identity": generation_identity,
        "sft_generation_identity": sft_generation_identity,
        "generation_protocol_parity": generation_protocol_parity,
        "critic_template": critic_template,
        "config_hash": _config_hash(args),
        "code_revision": _git_revision(),
        "row_keys": [list(provenance_key(record)) for record in selected],
    }
    preflight = build_functional_report(
        identity_rows=identity_rows,
        functional_rows=[],
        identity_tolerances=tolerances,
        metadata=metadata,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    if not preflight["gate"]["identity_passed"]:
        _release_target(model)
        return preflight, 2

    entries, shuffle_stats = build_variant_entries(
        selected,
        sources,
        predictions,
        mean_activation,
    )
    functional_rows = run_functional_pass(
        model=model,
        boundary_module=boundary_module,
        entries=entries,
        original_logits=original_logits,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
    )
    _release_target(model)
    metadata["within_document_shuffle"] = shuffle_stats
    return (
        build_functional_report(
            identity_rows=identity_rows,
            functional_rows=functional_rows,
            identity_tolerances=tolerances,
            metadata=metadata,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        0,
    )


def _add_bool_optional(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
) -> None:
    destination = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=destination, action="store_true")
    group.add_argument("--no-" + name[2:], dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--sft-generated-jsonl", type=Path)
    parser.add_argument("--ar-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-base-parquet", type=Path, required=True)
    parser.add_argument("--mean-activation-parquet", type=Path)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--validation-limit", type=int, required=True)
    parser.add_argument("--test-limit", type=int, required=True)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("row_order", "longest_prefix"),
        default="row_order",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--ar-batch-size", type=int)
    parser.add_argument("--ar-max-length", type=int)
    parser.add_argument("--ar-torch-dtype", default="bfloat16")
    parser.add_argument("--ar-device-map", default="auto")
    parser.add_argument("--target-torch-dtype", default="bfloat16")
    parser.add_argument("--target-device-map", default="auto")
    parser.add_argument("--target-revision")
    parser.add_argument("--identity-relative-l2", type=float, required=True)
    parser.add_argument("--identity-max-abs", type=float, required=True)
    parser.add_argument("--identity-one-minus-cos", type=float, required=True)
    parser.add_argument("--control", default="real")
    parser.add_argument("--min-independent-families", type=int, default=1)
    parser.add_argument("--critic-template")
    parser.add_argument("--critic-template-source", type=Path)
    parser.add_argument(
        "--generated-text-fallback",
        choices=("empty", "raw"),
        default="empty",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-batch-size", type=int, default=4_096)
    _add_bool_optional(parser, "--target-local-files-only", default=True)
    _add_bool_optional(parser, "--target-trust-remote-code", default=True)
    _add_bool_optional(parser, "--require-generation-identity", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.boundary <= 0:
        raise FunctionalEvaluationError("boundary must be positive")
    limits = {"validation": args.validation_limit, "test": args.test_limit}
    if any(limits[split] <= 0 for split in args.eval_splits):
        raise FunctionalEvaluationError("selected split limits must be positive")
    if args.batch_size <= 0 or (args.ar_batch_size is not None and args.ar_batch_size <= 0):
        raise FunctionalEvaluationError("batch sizes must be positive")
    if args.bootstrap_resamples <= 0 or args.source_batch_size <= 0:
        raise FunctionalEvaluationError("bootstrap and source batch sizes must be positive")
    if args.min_independent_families <= 0:
        raise FunctionalEvaluationError("min_independent_families must be positive")
    report, return_code = run(args)
    write_json(args.report_json, report)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
