#!/usr/bin/env python3
"""Compare stored R33 activations with full-model and extraction-path forwards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_prefix_activation_extract import (  # noqa: E402
    add_execution_profile_arguments,
    configure_extraction_execution,
    validate_execution_profile,
)


def inference_call(torch_module: Any, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call an extraction forward under the same no-grad contract as extraction."""

    with torch_module.no_grad():
        return function(*args, **kwargs)


def assess_identity_rows(
    rows: Sequence[dict[str, Any]],
    *,
    comparison: str,
    max_relative_l2: float,
    max_abs: float,
    max_one_minus_cos: float,
) -> dict[str, Any]:
    """Evaluate a live-vs-stored activation comparison against fixed tolerances."""

    if not rows:
        raise ValueError("activation fidelity assessment requires at least one row")
    thresholds = {
        "max_relative_l2": float(max_relative_l2),
        "max_abs": float(max_abs),
        "max_one_minus_cos": float(max_one_minus_cos),
    }
    if any(value < 0 for value in thresholds.values()):
        raise ValueError("activation fidelity tolerances must be nonnegative")
    metrics = [row[comparison] for row in rows]
    violations = [
        index
        for index, metric in enumerate(metrics)
        if float(metric["relative_l2"]) > thresholds["max_relative_l2"]
        or float(metric["max_abs"]) > thresholds["max_abs"]
        or float(metric["one_minus_cos"]) > thresholds["max_one_minus_cos"]
    ]
    return {
        "comparison": comparison,
        "row_count": len(rows),
        "passed": not violations,
        "violating_rows": len(violations),
        "violating_row_indices": violations,
        "thresholds": thresholds,
        "max_observed_relative_l2": max(float(item["relative_l2"]) for item in metrics),
        "max_observed_abs": max(float(item["max_abs"]) for item in metrics),
        "max_observed_one_minus_cos": max(
            float(item["one_minus_cos"]) for item in metrics
        ),
    }


def summarize_activation_fidelity(
    *,
    live_values: Any,
    stored_values: Any,
    train_mean: Any,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Quantify the metric floor introduced by stored activation vectors."""

    import numpy as np

    from nano_eval_core import activation_reconstruction_metrics

    live = np.asarray(live_values, dtype=np.float64)
    stored = np.asarray(stored_values, dtype=np.float64)
    metrics = activation_reconstruction_metrics(
        stored,
        live,
        train_mean=np.asarray(train_mean, dtype=np.float64),
        eps=eps,
    )
    difference = live - stored
    absolute_l2 = np.linalg.norm(difference, axis=1)
    live_norms = np.linalg.norm(live, axis=1)
    stored_norms = np.linalg.norm(stored, axis=1)
    relative_l2 = absolute_l2 / np.maximum(live_norms, eps)
    cosine = np.sum(live * stored, axis=1) / (
        np.maximum(live_norms, eps) * np.maximum(stored_norms, eps)
    )
    valid_stored_norms = stored_norms > eps
    norm_ratio = (
        float(np.mean(live_norms[valid_stored_norms] / stored_norms[valid_stored_norms]))
        if np.any(valid_stored_norms)
        else None
    )
    scalar_metrics = {
        name: value
        for name, value in metrics.items()
        if not name.startswith("rowwise_") and name != "normalized_mse"
    }
    return {
        "row_count": int(live.shape[0]),
        "absolute_l2_mean": float(absolute_l2.mean()),
        "absolute_l2_max": float(absolute_l2.max()),
        "relative_l2_mean": float(relative_l2.mean()),
        "relative_l2_max": float(relative_l2.max()),
        "cosine_agreement_mean": float(cosine.mean()),
        "cosine_agreement_min": float(cosine.min()),
        "live_over_stored_norm_ratio_mean": norm_ratio,
        "stored_as_prediction_vs_live_target": scalar_metrics,
    }


def build_activation_fidelity_manifest(
    *,
    generated_jsonl: Path,
    source_base_parquet: Path,
    mean_activation_parquet: Path,
    extraction_source_parquet: Path | None,
    content_family_manifest: Path | None,
    target_model: str,
    target_model_fingerprint: str | None,
    target_revision: str | None,
    boundary: int,
    target_torch_dtype: str,
    selection_strategy: str,
    selection_seed: int,
    sample_identities: Sequence[tuple[Any, ...]],
    code_paths: Sequence[Path],
    publication_mode: bool,
    execution_profile: dict[str, Any],
) -> dict[str, Any]:
    """Bind an activation-fidelity result to all material inputs."""

    from nano_source_provenance import sha256_file

    if publication_mode and not str(target_model_fingerprint or "").strip():
        raise ValueError("publication activation fidelity requires checkpoint fingerprint")

    def file_identity(path: Path) -> dict[str, Any]:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ValueError(f"activation fidelity input is not a file: {resolved}")
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size_bytes": int(stat.st_size),
            "sha256": sha256_file(resolved),
        }

    input_paths = {
        "generated_jsonl": Path(generated_jsonl),
        "source_base_parquet": Path(source_base_parquet),
        "mean_activation_parquet": Path(mean_activation_parquet),
    }
    if extraction_source_parquet is not None:
        input_paths["extraction_source_parquet"] = Path(extraction_source_parquet)
    if content_family_manifest is not None:
        input_paths["content_family_manifest"] = Path(content_family_manifest)
    inputs = {name: file_identity(path) for name, path in input_paths.items()}
    code = [file_identity(Path(path)) for path in code_paths]
    identities = [list(identity) for identity in sample_identities]
    identities_payload = json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    complete = bool(
        str(target_model_fingerprint or "").strip()
        and inputs
        and code
        and identities
        and execution_profile
    )
    return {
        "schema_version": "nano_activation_fidelity_manifest.v1",
        "publication_complete": complete,
        "inputs": inputs,
        "activation_extraction": {
            "checkpoint": {
                "reference": str(target_model),
                "revision": target_revision,
                "fingerprint": target_model_fingerprint,
            },
            "boundary": int(boundary),
            "dtype": str(target_torch_dtype),
            "execution_profile": dict(execution_profile),
            "code": code,
        },
        "selection": {
            "strategy": str(selection_strategy),
            "seed": int(selection_seed),
            "sample_identities": identities,
            "sample_identities_sha256": hashlib.sha256(identities_payload).hexdigest(),
        },
    }


def build_identity_comparisons(
    *,
    keys: Sequence[tuple[Any, ...]],
    prefix_lengths: Sequence[int],
    stored: Sequence[Any],
    full_forward: Sequence[Any],
    extraction_forward: Sequence[Any],
    metric_fn: Callable[[Any, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = {
        len(keys),
        len(prefix_lengths),
        len(stored),
        len(full_forward),
        len(extraction_forward),
    }
    if len(counts) != 1:
        raise ValueError("identity methods must have the same row count")

    rows = []
    for index, key in enumerate(keys):
        rows.append(
            {
                "provenance_key": list(key),
                "prefix_length": int(prefix_lengths[index]),
                "full_vs_stored": metric_fn(full_forward[index], stored[index]),
                "extraction_vs_stored": metric_fn(
                    extraction_forward[index], stored[index]
                ),
                "full_vs_extraction": metric_fn(
                    full_forward[index], extraction_forward[index]
                ),
            }
        )
    return rows


def original_batch_starts(
    groups: Sequence[dict[str, Any]],
    *,
    target_doc_ids: set[str],
    batch_size: int,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    index_by_doc = {str(group["doc_id"]): index for index, group in enumerate(groups)}
    missing = sorted(target_doc_ids - set(index_by_doc))
    if missing:
        raise ValueError(f"missing target documents from extraction source: {missing}")
    return sorted(
        {
            (index_by_doc[doc_id] // batch_size) * batch_size
            for doc_id in target_doc_ids
        }
    )


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    requested_execution_profile = validate_execution_profile(args)
    if requested_execution_profile["cublas_workspace_config"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = requested_execution_profile[
            "cublas_workspace_config"
        ]
    import torch

    execution_profile = configure_extraction_execution(args, torch)

    from nano_extraction_identity import tensor_metrics
    from nano_functional_eval_data import (
        attach_content_family_ids,
        load_content_family_manifest,
        read_generated_jsonl,
        select_exact_split_rows,
        source_mean_activation,
    )
    from nano_introspection import resolve_nano_module_paths
    from nano_prefix_activation_extract import _import_runtime_helpers
    from nano_prefix_activation_extract import (
        collect_source_records,
        group_records_by_doc,
        plan_group_batch,
    )
    from nano_r33_functional_runtime import (
        _capture_boundary_forward,
        _module_device,
        _pad_prefixes,
        load_target_model,
    )
    from nano_r33_source_rows import (
        provenance_key,
        resolve_source_rows,
    )

    helpers = _import_runtime_helpers()
    generated_rows = read_generated_jsonl(args.generated_jsonl)
    if args.content_family_manifest is not None:
        attach_content_family_ids(
            generated_rows,
            load_content_family_manifest(args.content_family_manifest),
        )
    selected = select_exact_split_rows(
        generated_rows,
        args.validation_limit,
        args.test_limit,
        eval_splits=args.eval_splits,
        selection_strategy=args.selection_strategy,
        selection_seed=args.selection_seed,
    )
    source_map = resolve_source_rows(
        args.source_base_parquet,
        selected,
        batch_size=args.source_batch_size,
    )
    sources = [source_map[provenance_key(record)] for record in selected]
    prefixes = [
        [int(token) for token in source["token_ids_prefix"]] for source in sources
    ]
    keys = [provenance_key(record) for record in selected]

    extraction_groups = None
    extraction_starts: list[int] = []
    if args.extraction_source_parquet is not None:
        print(json.dumps({"stage": "reconstruct_original_batches"}), flush=True)
        extraction_records, _ = collect_source_records(
            args.extraction_source_parquet,
            batch_size=args.source_batch_size,
        )
        extraction_groups = group_records_by_doc(extraction_records)
        extraction_starts = original_batch_starts(
            extraction_groups,
            target_doc_ids={str(source["doc_id"]) for source in sources},
            batch_size=args.extraction_batch_size,
        )

    print(
        json.dumps(
            {
                "stage": "load_target_model",
                "row_count": len(selected),
                "loader": args.model_loader,
            }
        ),
        flush=True,
    )
    if args.model_loader == "extraction":
        model_args = SimpleNamespace(
            model_id=args.target_model,
            model_revision=args.target_revision,
            tokenizer_revision=args.target_revision,
            torch_dtype=args.target_torch_dtype,
            trust_remote_code=args.target_trust_remote_code,
            local_files_only=args.target_local_files_only,
            device_map=args.target_device_map,
            attn_implementation=None,
            load_mode="full",
        )
        config, config_error = helpers["load_config_from_args"](model_args)
        if config_error is not None:
            raise RuntimeError(config_error)
        model = helpers["load_model_from_args"](model_args, config)
    else:
        model_args = SimpleNamespace(
            target_model=args.target_model,
            target_torch_dtype=args.target_torch_dtype,
            target_trust_remote_code=args.target_trust_remote_code,
            target_local_files_only=args.target_local_files_only,
            target_revision=args.target_revision,
            target_device_map=args.target_device_map,
        )
        model = load_target_model(model_args)
    try:
        resolved = resolve_nano_module_paths(model)
        layers = resolved["layers"].obj
        if layers is None or not 1 <= args.boundary <= len(layers):
            raise ValueError(
                f"boundary {args.boundary} is invalid for "
                f"{0 if layers is None else len(layers)} layers"
            )
        boundary_module = layers[args.boundary - 1]
        start_device = _module_device(model.get_input_embeddings())
        pad_token_id = getattr(model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "eos_token_id", 0)
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]

        extraction_geometry: list[dict[str, Any]] = []
        original_by_key: dict[tuple[str, int], Any] = {}
        if extraction_groups is not None:
            target_keys = {
                (str(source["doc_id"]), int(source["n_raw_tokens"]))
                for source in sources
            }
            for batch_start in extraction_starts:
                batch_groups = extraction_groups[
                    batch_start : batch_start + args.extraction_batch_size
                ]
                planned = plan_group_batch(
                    batch_groups,
                    pad_token_id=int(pad_token_id or 0),
                )
                original_input_ids = torch.tensor(
                    planned["input_ids"],
                    dtype=torch.long,
                    device=start_device,
                )
                original_attention_mask = torch.tensor(
                    planned["attention_mask"],
                    dtype=torch.long,
                    device=start_device,
                )
                print(
                    json.dumps(
                        {
                            "stage": "original_extraction_forward",
                            "batch_start": batch_start,
                            "docs": [group["doc_id"] for group in batch_groups],
                            "max_length": int(original_input_ids.shape[1]),
                        }
                    ),
                    flush=True,
                )
                original_map = inference_call(
                    torch,
                    helpers["_forward_selected_boundaries"],
                    helpers=helpers,
                    model=model,
                    input_ids=original_input_ids,
                    attention_mask=original_attention_mask,
                    layers=[args.boundary],
                    selected_positions=planned["selected_positions"],
                )[args.boundary]
                for batch_index, record in planned["batch_records"]:
                    key = (str(record["doc_id"]), int(record["n_raw_tokens"]))
                    if key not in target_keys:
                        continue
                    position = int(record["selected_position"])
                    original_by_key[key] = torch.tensor(
                        original_map[(batch_index, position)],
                        dtype=torch.float32,
                    )
                extraction_geometry.append(
                    {
                        "batch_start": batch_start,
                        "docs": [
                            {
                                "doc_id": str(group["doc_id"]),
                                "record_count": len(group["records"]),
                                "longest_prefix": len(group["token_ids"]),
                            }
                            for group in batch_groups
                        ],
                        "max_length": int(original_input_ids.shape[1]),
                    }
                )

        full_values: list[Any] = []
        repeated_values: list[Any] | None = [] if args.repeat_full_forward else None
        extraction_values: list[Any] = []
        for start in range(0, len(selected), args.batch_size):
            batch_prefixes = prefixes[start : start + args.batch_size]
            input_ids, attention_mask, positions = _pad_prefixes(
                batch_prefixes,
                pad_token_id=int(pad_token_id or 0),
                device=start_device,
            )
            print(
                json.dumps(
                    {
                        "stage": "full_model_forward",
                        "row_start": start,
                        "row_count": len(batch_prefixes),
                    }
                ),
                flush=True,
            )
            batch_full_values, full_logits = _capture_boundary_forward(
                model,
                boundary_module,
                input_ids,
                attention_mask,
                positions,
            )
            full_values.extend(
                value.detach().float().cpu() for value in batch_full_values
            )
            del full_logits, batch_full_values

            if repeated_values is not None:
                print(
                    json.dumps(
                        {
                            "stage": "repeat_full_model_forward",
                            "row_start": start,
                            "row_count": len(batch_prefixes),
                        }
                    ),
                    flush=True,
                )
                batch_repeated_values, repeated_logits = _capture_boundary_forward(
                    model,
                    boundary_module,
                    input_ids,
                    attention_mask,
                    positions,
                )
                repeated_values.extend(
                    value.detach().float().cpu()
                    for value in batch_repeated_values
                )
                del repeated_logits, batch_repeated_values

            print(
                json.dumps(
                    {
                        "stage": "extraction_prefix_forward",
                        "row_start": start,
                        "row_count": len(batch_prefixes),
                    }
                ),
                flush=True,
            )
            extraction_map = inference_call(
                torch,
                helpers["_forward_selected_boundaries"],
                helpers=helpers,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layers=[args.boundary],
                selected_positions=[
                    (index, int(positions[index].item()))
                    for index in range(len(batch_prefixes))
                ],
            )[args.boundary]
            extraction_values.extend(
                torch.tensor(
                    extraction_map[(index, int(positions[index].item()))],
                    dtype=torch.float32,
                ).cpu()
                for index in range(len(batch_prefixes))
            )
            del input_ids, attention_mask, positions, extraction_map

        stored_values = [
            torch.tensor(
                source["activation_vector"],
                dtype=torch.float32,
            )
            for source in sources
        ]
        train_mean = source_mean_activation(
            args.mean_activation_parquet or args.source_base_parquet,
            batch_size=args.source_batch_size,
        )
        full_matrix = torch.stack(full_values).detach().float().cpu().numpy()
        extraction_matrix = torch.stack(extraction_values).detach().float().cpu().numpy()
        stored_matrix = torch.stack(stored_values).detach().float().cpu().numpy()
        activation_fidelity = {
            "full_forward_vs_stored": summarize_activation_fidelity(
                live_values=full_matrix,
                stored_values=stored_matrix,
                train_mean=train_mean,
            ),
            "extraction_forward_vs_stored": summarize_activation_fidelity(
                live_values=extraction_matrix,
                stored_values=stored_matrix,
                train_mean=train_mean,
            ),
        }
        rows = build_identity_comparisons(
            keys=keys,
            prefix_lengths=[len(prefix) for prefix in prefixes],
            stored=stored_values,
            full_forward=full_values,
            extraction_forward=extraction_values,
            metric_fn=tensor_metrics,
        )
        if repeated_values is not None:
            for index, row in enumerate(rows):
                row["full_repeat_vs_full"] = tensor_metrics(
                    repeated_values[index], full_values[index]
                )
        if extraction_groups is not None:
            original_values = [
                original_by_key[
                    (str(source["doc_id"]), int(source["n_raw_tokens"]))
                ].detach().float().cpu()
                for source in sources
            ]
            for index, row in enumerate(rows):
                row["original_extraction_vs_stored"] = tensor_metrics(
                    original_values[index], stored_values[index]
                )
                row["full_vs_original_extraction"] = tensor_metrics(
                    full_values[index], original_values[index]
                )
                row["current_vs_original_extraction"] = tensor_metrics(
                    extraction_values[index], original_values[index]
                )
        assessments = {
            "full_forward_vs_stored": assess_identity_rows(
                rows,
                comparison="full_vs_stored",
                max_relative_l2=args.fidelity_max_relative_l2,
                max_abs=args.fidelity_max_abs,
                max_one_minus_cos=args.fidelity_max_one_minus_cos,
            )
        }
        primary_assessment = "full_forward_vs_stored"
        if extraction_groups is not None:
            assessments["original_extraction_vs_stored"] = assess_identity_rows(
                rows,
                comparison="original_extraction_vs_stored",
                max_relative_l2=args.fidelity_max_relative_l2,
                max_abs=args.fidelity_max_abs,
                max_one_minus_cos=args.fidelity_max_one_minus_cos,
            )
            primary_assessment = "original_extraction_vs_stored"
        data_manifest = build_activation_fidelity_manifest(
            generated_jsonl=args.generated_jsonl,
            source_base_parquet=args.source_base_parquet,
            mean_activation_parquet=(
                args.mean_activation_parquet or args.source_base_parquet
            ),
            extraction_source_parquet=args.extraction_source_parquet,
            content_family_manifest=args.content_family_manifest,
            target_model=args.target_model,
            target_model_fingerprint=args.target_model_fingerprint,
            target_revision=args.target_revision,
            boundary=args.boundary,
            target_torch_dtype=args.target_torch_dtype,
            selection_strategy=args.selection_strategy,
            selection_seed=args.selection_seed,
            sample_identities=keys,
            code_paths=[
                Path(__file__),
                SCRIPT_DIR / "nano_extraction_identity.py",
                SCRIPT_DIR / "nano_prefix_activation_extract.py",
                SCRIPT_DIR / "nano_r33_functional_runtime.py",
                SCRIPT_DIR / "nano_r33_source_rows.py",
            ],
            publication_mode=args.publication_mode,
            execution_profile=execution_profile,
        )
        report = {
            "schema_version": "nano_r33_activation_identity_diagnostic.v2",
            "metadata": {
                "generated_jsonl": str(args.generated_jsonl),
                "source_base_parquet": str(args.source_base_parquet),
                "target_model": args.target_model,
                "target_torch_dtype": args.target_torch_dtype,
                "target_device_map": args.target_device_map,
                "model_loader": args.model_loader,
                "repeat_full_forward": args.repeat_full_forward,
                "boundary": args.boundary,
                "sample_row_count": len(selected),
                "batch_size": args.batch_size,
                "eval_splits": list(args.eval_splits),
                "selection_strategy": args.selection_strategy,
                "selection_seed": args.selection_seed,
                "publication_mode": args.publication_mode,
                "execution_profile": execution_profile,
                "content_family_manifest": str(args.content_family_manifest)
                if args.content_family_manifest is not None
                else None,
                "target_model_fingerprint": args.target_model_fingerprint,
                "fidelity_tolerances": {
                    "max_relative_l2": args.fidelity_max_relative_l2,
                    "max_abs": args.fidelity_max_abs,
                    "max_one_minus_cos": args.fidelity_max_one_minus_cos,
                },
                "mean_activation_parquet": str(
                    args.mean_activation_parquet or args.source_base_parquet
                ),
                "extraction_source_parquet": str(args.extraction_source_parquet)
                if args.extraction_source_parquet is not None
                else None,
                "extraction_batch_size": args.extraction_batch_size,
                "original_extraction_geometry": extraction_geometry,
                "module_paths": {
                    name: value.path for name, value in resolved.items()
                },
            },
            "data_manifest": data_manifest,
            "activation_fidelity": activation_fidelity,
            "fidelity_assessments": assessments,
            "primary_fidelity_assessment": primary_assessment,
            "publication_ready": bool(
                data_manifest["publication_complete"]
                and assessments[primary_assessment]["passed"]
            ),
            "rows": rows,
        }
    finally:
        _release_model(model)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--source-base-parquet", type=Path, required=True)
    parser.add_argument("--mean-activation-parquet", type=Path)
    parser.add_argument("--content-family-manifest", type=Path)
    parser.add_argument("--extraction-source-parquet", type=Path)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-model-fingerprint")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--validation-limit", type=int, required=True)
    parser.add_argument("--test-limit", type=int, required=True)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--source-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extraction-batch-size", type=int, default=2)
    parser.add_argument(
        "--selection-strategy",
        choices=("row_order", "longest_prefix", "family_stratified"),
        default="row_order",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--publication-mode", action="store_true")
    parser.add_argument("--fidelity-max-relative-l2", type=float, default=0.01)
    parser.add_argument("--fidelity-max-abs", type=float, default=0.01)
    parser.add_argument("--fidelity-max-one-minus-cos", type=float, default=0.0001)
    parser.add_argument("--target-torch-dtype", default="bfloat16")
    parser.add_argument("--target-device-map", default="auto")
    parser.add_argument("--target-revision")
    parser.add_argument(
        "--model-loader",
        choices=("target", "extraction"),
        default="target",
    )
    parser.add_argument("--repeat-full-forward", action="store_true")
    parser.add_argument(
        "--target-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    add_execution_profile_arguments(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    validate_execution_profile(args)
    if (
        args.boundary <= 0
        or args.validation_limit <= 0
        or args.test_limit <= 0
        or args.source_batch_size <= 0
        or args.batch_size <= 0
        or args.extraction_batch_size <= 0
    ):
        raise ValueError("boundary, split limits, and batch sizes must be positive")
    if args.publication_mode:
        if not str(args.target_model_fingerprint or "").strip():
            raise ValueError("publication activation fidelity requires checkpoint fingerprint")
        if args.selection_strategy != "family_stratified":
            raise ValueError(
                "publication activation fidelity requires family_stratified selection"
            )
        if args.content_family_manifest is None:
            raise ValueError(
                "publication activation fidelity requires content-family manifest"
            )
    if min(
        args.fidelity_max_relative_l2,
        args.fidelity_max_abs,
        args.fidelity_max_one_minus_cos,
    ) < 0:
        raise ValueError("activation fidelity tolerances must be nonnegative")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    report = run(args)
    summary = {
        row["provenance_key"][-1]: {
            name: metrics["relative_l2"]
            for name, metrics in row.items()
            if name.endswith("_stored")
            or name in {
                "full_vs_extraction",
                "full_vs_original_extraction",
                "current_vs_original_extraction",
                "full_repeat_vs_full",
            }
        }
        for row in report["rows"]
    }
    print(json.dumps({"report": str(args.report_json), "relative_l2": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
