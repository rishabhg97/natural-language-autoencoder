#!/usr/bin/env python3
"""Nano AR data-signal gate before scaled AR SFT.

This diagnostic keeps Nano frozen and compares prompt sources for the same
target residuals:

* teacher explanation critic prompts
* shuffled teacher explanation prompts
* blank/generic critic prompts
* source text inside the critic template
* raw source prefix prompts

The raw source prefix is an oracle/control path, not the AR contract. AR must
learn from explanation text z, not source context x. If the source path works
but teacher explanations do not beat blank/shuffled/mean controls, scaling more
teacher rows is unlikely to fix the current heldout failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_ar_frozen_baseline import (  # noqa: E402
    DEFAULT_CRITIC_TEMPLATE,
    ValueHead,
    centered_raw_diagnostics,
    mean_target_metrics,
    split_indices,
    split_metadata,
    take_rows,
    train_value_head,
    vector_metrics,
)
from nano_extraction_identity import parse_boundaries, prefix_forward_to_R_b  # noqa: E402
from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_ROOT,
    add_bool_optional_arg,
    build_metadata_record,
    classify_blocker,
    get_config_value,
    json_safe,
    load_config_from_args,
    load_model_from_args,
    load_tokenizer_from_args,
    make_run_dir,
    write_json,
)
from nano_wandb import add_wandb_args, init_wandb  # noqa: E402


VARIANT_ORDER = (
    "teacher",
    "teacher_shuffled",
    "blank",
    "generic",
    "source_context",
    "source_raw",
)


class SignalRow(NamedTuple):
    row_index: int
    record_id: str
    boundary_b: int
    target: list[float]
    teacher_prompt: str
    api_explanation: str | None
    source_text: str | None
    source_token_ids: list[int] | None
    metadata: dict[str, Any]


class VariantItem(NamedTuple):
    variant: str
    row: SignalRow
    text: str | None
    token_ids: list[int] | None
    source_row_index: int
    provenance: str


def format_critic_prompt(content: str, critic_template: str) -> str:
    if "{explanation}" not in critic_template:
        raise ValueError("critic_template must contain {explanation}")
    return critic_template.format(explanation=content)


def _as_float_list(value: Any, row_idx: int) -> list[float]:
    if value is None:
        raise ValueError(f"row {row_idx} activation_vector is null")
    return [float(item) for item in value]


def _as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        return None
    return [int(item) for item in value]


def load_signal_rows(
    parquet_path: Path,
    *,
    boundaries: list[int],
    max_records: int,
    source_column: str,
    source_token_ids_column: str,
) -> tuple[list[SignalRow], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for --ar-sft-parquet") from exc

    table = pq.read_table(parquet_path)
    names = set(table.column_names)
    required = {"prompt", "activation_vector"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {missing}")

    requested = set(int(item) for item in boundaries)
    if "activation_layer" not in names and len(requested) != 1:
        raise ValueError("parquet has no activation_layer column; pass exactly one --boundaries value")

    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    rows: list[SignalRow] = []
    for row_idx in range(table.num_rows):
        boundary_b = int(cols["activation_layer"][row_idx]) if "activation_layer" in cols else boundaries[0]
        if boundary_b not in requested:
            continue
        prompt = cols["prompt"][row_idx]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"row {row_idx} prompt must be non-empty string")
        doc_id = cols.get("doc_id", [None] * table.num_rows)[row_idx] if "doc_id" in cols else None
        source_text = cols.get(source_column, [None] * table.num_rows)[row_idx] if source_column in cols else None
        if source_text is not None and not isinstance(source_text, str):
            source_text = str(source_text)
        api_explanation = cols.get("api_explanation", [None] * table.num_rows)[row_idx] if "api_explanation" in cols else None
        if api_explanation is not None and not isinstance(api_explanation, str):
            api_explanation = str(api_explanation)
        source_token_ids = _as_int_list(
            cols.get(source_token_ids_column, [None] * table.num_rows)[row_idx]
            if source_token_ids_column in cols
            else None
        )
        rows.append(
            SignalRow(
                row_index=row_idx,
                record_id=str(doc_id) if doc_id is not None else f"row_{row_idx}",
                boundary_b=boundary_b,
                target=_as_float_list(cols["activation_vector"][row_idx], row_idx),
                teacher_prompt=prompt,
                api_explanation=api_explanation,
                source_text=source_text,
                source_token_ids=source_token_ids,
                metadata={
                    "row_index": row_idx,
                    "doc_id": doc_id,
                    "n_raw_tokens": cols.get("n_raw_tokens", [None] * table.num_rows)[row_idx]
                    if "n_raw_tokens" in cols
                    else None,
                    "token_position": cols.get("token_position", [None] * table.num_rows)[row_idx]
                    if "token_position" in cols
                    else None,
                    "token_id": cols.get("token_id", [None] * table.num_rows)[row_idx] if "token_id" in cols else None,
                    "token_text": cols.get("token_text", [None] * table.num_rows)[row_idx]
                    if "token_text" in cols
                    else None,
                },
            )
        )
        if len(rows) >= max_records:
            break

    if not rows:
        raise ValueError(f"no rows matched boundaries {boundaries} in {parquet_path}")

    provenance = {
        "row_count": len(rows),
        "columns": table.column_names,
        "source_column": source_column if source_column in names else None,
        "source_token_ids_column": source_token_ids_column if source_token_ids_column in names else None,
        "source_text_count": int(sum(1 for row in rows if row.source_text)),
        "source_token_ids_count": int(sum(1 for row in rows if row.source_token_ids)),
        "api_explanation_count": int(sum(1 for row in rows if row.api_explanation)),
    }
    provenance["exact_token_prefix_fraction"] = provenance["source_token_ids_count"] / max(1, len(rows))
    return rows, provenance


def shuffled_indices(count: int, seed: int) -> list[int]:
    if count <= 1:
        return list(range(count))
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    for idx, source_idx in enumerate(indices):
        if source_idx == idx:
            swap_idx = (idx + 1) % count
            indices[idx], indices[swap_idx] = indices[swap_idx], indices[idx]
    return indices


def build_variant_items(
    rows: list[SignalRow],
    *,
    variant: str,
    critic_template: str,
    generic_explanation: str,
    seed: int,
) -> list[VariantItem]:
    if variant not in VARIANT_ORDER:
        raise ValueError(f"unknown variant {variant!r}")
    shuffled = shuffled_indices(len(rows), seed)
    items: list[VariantItem] = []
    for idx, row in enumerate(rows):
        source_idx = idx
        text: str | None
        token_ids: list[int] | None = None
        provenance: str
        if variant == "teacher":
            text = row.teacher_prompt
            provenance = "teacher_prompt"
        elif variant == "teacher_shuffled":
            source_idx = shuffled[idx]
            text = rows[source_idx].teacher_prompt
            provenance = "teacher_prompt_shuffled"
        elif variant == "blank":
            text = format_critic_prompt("", critic_template)
            provenance = "blank_critic_prompt"
        elif variant == "generic":
            text = format_critic_prompt(generic_explanation, critic_template)
            provenance = "generic_critic_prompt"
        elif variant == "source_context":
            if not row.source_text:
                raise ValueError("source_context variant requires source text")
            text = format_critic_prompt(row.source_text, critic_template)
            provenance = "source_text_inside_critic_prompt"
        elif variant == "source_raw":
            if row.source_token_ids:
                text = None
                token_ids = row.source_token_ids
                provenance = "source_token_ids_prefix"
            elif row.source_text:
                text = row.source_text
                provenance = "source_text_retokenized"
            else:
                raise ValueError("source_raw variant requires source token ids or source text")
        items.append(
            VariantItem(
                variant=variant,
                row=row,
                text=text,
                token_ids=token_ids,
                source_row_index=source_idx,
                provenance=provenance,
            )
        )
    return items


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def encode_text_no_truncate(tokenizer: Any, text: str, max_length: int | None) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(item) for item in ids]
    if max_length is not None and len(ids) > max_length:
        raise ValueError(
            f"prompt token_count={len(ids)} exceeds max_length={max_length}; do not truncate because the final token is extracted"
        )
    if not ids:
        raise ValueError("prompt tokenized to zero tokens")
    return ids


def check_token_ids_no_truncate(token_ids: list[int], max_length: int | None) -> list[int]:
    if not token_ids:
        raise ValueError("source token_ids_prefix is empty")
    ids = [int(item) for item in token_ids]
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f"source token_ids_prefix length={len(ids)} exceeds max_length={max_length}")
    return ids


def pad_token_rows(tokenizer: Any, token_rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(row) for row in token_rows]
    max_len = max(lengths)
    input_ids = torch.full((len(token_rows), max_len), _pad_token_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(token_rows), max_len), dtype=torch.long)
    for row_idx, row in enumerate(token_rows):
        row_len = lengths[row_idx]
        input_ids[row_idx, :row_len] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_idx, :row_len] = 1
    return input_ids, attention_mask, lengths


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def select_token_vectors_by_lengths(hidden: torch.Tensor, lengths: list[int], tau: int) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for row_idx, token_count in enumerate(lengths):
        resolved = tau if tau >= 0 else token_count + tau
        if not 0 <= resolved < token_count:
            raise ValueError(f"tau={tau} resolves to {resolved} for token_count={token_count}")
        vectors.append(hidden[row_idx, resolved])
    return torch.stack(vectors, dim=0)


def materialize_variant_features(
    *,
    model: Any,
    tokenizer: Any,
    items: list[VariantItem],
    boundary_b: int,
    max_length: int | None,
    batch_size: int,
    tau: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    features: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            token_rows = [
                check_token_ids_no_truncate(item.token_ids, max_length)
                if item.token_ids is not None
                else encode_text_no_truncate(tokenizer, item.text or "", max_length)
                for item in batch_items
            ]
            input_ids, attention_mask, lengths = pad_token_rows(tokenizer, token_rows)
            input_ids = input_ids.to(_model_start_device(model))
            attention_mask = attention_mask.to(input_ids.device)
            hidden = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b)
            batch_features = select_token_vectors_by_lengths(hidden, lengths, tau).detach().float().cpu()
            features.extend([row for row in batch_features])
            for item, token_ids, token_count in zip(batch_items, token_rows, lengths, strict=True):
                text = item.text if item.text is not None else ""
                records.append(
                    {
                        "variant": item.variant,
                        "record_id": item.row.record_id,
                        "row_index": item.row.row_index,
                        "source_row_index": item.source_row_index,
                        "provenance": item.provenance,
                        "token_count": int(token_count),
                        "token_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
                        "text_preview": text[:160] if text else None,
                    }
                )
    return torch.stack(features, dim=0), records


def variant_eval(
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    train_indices: list[int],
    eval_indices: list[int],
    max_steps: int,
    lr: float,
    weight_decay: float,
    log_every: int,
    train_targets_for_mean: torch.Tensor,
) -> dict[str, Any]:
    hidden_size = int(targets.shape[-1])
    head = ValueHead(hidden_size=hidden_size)
    train_features = take_rows(features, train_indices)
    train_targets = take_rows(targets, train_indices)
    eval_features = take_rows(features, eval_indices)
    eval_targets = take_rows(targets, eval_indices)
    feature_train = vector_metrics(train_features, train_targets)
    feature_eval = vector_metrics(eval_features, eval_targets)
    feature_train.update(centered_raw_diagnostics(train_features, train_targets, train_targets_for_mean))
    feature_eval.update(centered_raw_diagnostics(eval_features, eval_targets, train_targets_for_mean))

    before_train = vector_metrics(head(train_features.float()).detach(), train_targets)
    history = train_value_head(
        head,
        train_features,
        train_targets,
        max_steps=max_steps,
        lr=lr,
        weight_decay=weight_decay,
        log_every=log_every,
    )
    with torch.no_grad():
        train_pred = head(train_features.float()).detach().cpu()
        eval_pred = head(eval_features.float()).detach().cpu()
    after_train = vector_metrics(train_pred, train_targets)
    after_eval = vector_metrics(eval_pred, eval_targets)
    after_train.update(centered_raw_diagnostics(train_pred, train_targets, train_targets_for_mean))
    after_eval.update(centered_raw_diagnostics(eval_pred, eval_targets, train_targets_for_mean))
    mean_eval = mean_target_metrics(eval_targets, train_targets_for_mean)
    mean_train = mean_target_metrics(train_targets, train_targets_for_mean)
    return {
        "feature_train": feature_train,
        "feature_heldout": feature_eval,
        "head_train_before": before_train,
        "head_train_after": after_train,
        "head_heldout_after": after_eval,
        "mean_train": mean_train,
        "mean_heldout": mean_eval,
        "history": history,
        "train_loss_decreased": after_train["normalized_mse"] < before_train["normalized_mse"],
    }


def compare_teacher_controls(
    variants: dict[str, dict[str, Any]],
    *,
    mse_margin: float,
    cosine_margin: float,
    oracle_mse_threshold: float,
) -> dict[str, Any]:
    teacher = variants.get("teacher", {}).get("head_heldout_after")
    if teacher is None:
        return {"teacher_beats_controls": False, "reason": "missing teacher variant"}
    control_names = ["teacher_shuffled", "blank", "generic"]
    if "source_context" in variants:
        control_names.append("source_context")
    comparisons: dict[str, Any] = {}
    beats_all = True
    for name in control_names:
        control = variants.get(name, {}).get("head_heldout_after")
        if control is None:
            continue
        beats = (
            teacher["normalized_mse"] <= control["normalized_mse"] - mse_margin
            and teacher["cosine_mean"] >= control["cosine_mean"] + cosine_margin
        )
        comparisons[name] = {
            "beats": bool(beats),
            "control_normalized_mse": control["normalized_mse"],
            "control_cosine_mean": control["cosine_mean"],
        }
        beats_all = beats_all and bool(beats)
    mean = variants["teacher"]["mean_heldout"]
    teacher_vs_mean = teacher["normalized_mse"] <= mean["normalized_mse"] - mse_margin
    source_raw_feature = variants.get("source_raw", {}).get("feature_heldout")
    source_raw_oracle = (
        source_raw_feature is not None and source_raw_feature["normalized_mse"] <= oracle_mse_threshold
    )
    return {
        "teacher_beats_controls": bool(beats_all and teacher_vs_mean),
        "teacher_beats_mean": bool(teacher_vs_mean),
        "teacher_heldout_normalized_mse": teacher["normalized_mse"],
        "teacher_heldout_cosine": teacher["cosine_mean"],
        "mean_heldout_normalized_mse": mean["normalized_mse"],
        "source_raw_oracle_passed": bool(source_raw_oracle),
        "source_raw_feature_normalized_mse": source_raw_feature["normalized_mse"] if source_raw_feature else None,
        "source_raw_feature_cosine": source_raw_feature["cosine_mean"] if source_raw_feature else None,
        "control_comparisons": comparisons,
    }


def parse_variants(text: str) -> list[str]:
    variants = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANT_ORDER))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {unknown}; choices={VARIANT_ORDER}")
    return variants


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    parser.add_argument("--ar-sft-parquet", type=Path, required=True)
    parser.add_argument("--boundaries", type=parse_boundaries, default=[34])
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--split-strategy", choices=("sequential", "alternating", "random", "doc_random"), default="doc_random")
    parser.add_argument("--variants", type=parse_variants, default=list(VARIANT_ORDER))
    parser.add_argument("--critic-template", default=DEFAULT_CRITIC_TEMPLATE)
    parser.add_argument("--generic-explanation", default="The text contains ordinary information with several broad features.")
    parser.add_argument("--source-column", default="detokenized_text_truncated")
    parser.add_argument("--source-token-ids-column", default="token_ids_prefix")
    parser.add_argument("--prompt-max-length", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--tau", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--mse-margin", type=float, default=0.05)
    parser.add_argument("--cosine-margin", type=float, default=0.02)
    parser.add_argument("--oracle-mse-threshold", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    add_wandb_args(parser)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    output_path = run_dir / "ar_signal_gate.json"
    payload: dict[str, Any] = {
        "schema_version": "nano_ar_signal_gate.v1",
        "run_dir": str(run_dir),
        "data_source": {
            "path": str(args.ar_sft_parquet),
            "source_column": args.source_column,
            "source_token_ids_column": args.source_token_ids_column,
        },
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "boundaries": args.boundaries,
        "variants_requested": args.variants,
        "max_records": args.max_records,
        "train_fraction": args.train_fraction,
        "split_strategy": args.split_strategy,
        "variant_results": {},
        "comparison": {},
        "passed": False,
        "scientific_passed": False,
        "blockers": [],
    }
    tracker = init_wandb(
        args,
        run_dir=run_dir,
        job_type="ar_signal_gate",
        config=json_safe({"args": vars(args), "run_dir": run_dir}),
    )
    payload["wandb"] = tracker.metadata

    if torch is None:
        payload["blockers"] = [{"kind": "environment", "label": "torch import", "error": "PyTorch is required"}]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2
    if len(args.boundaries) != 1:
        payload["blockers"] = [{"kind": "configuration", "label": "boundaries", "error": "signal gate accepts exactly one boundary"}]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    blockers: list[dict[str, str]] = []
    boundary_b = int(args.boundaries[0])
    try:
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
        rows, provenance = load_signal_rows(
            args.ar_sft_parquet,
            boundaries=[boundary_b],
            max_records=args.max_records,
            source_column=args.source_column,
            source_token_ids_column=args.source_token_ids_column,
        )
        payload["data_source"]["provenance"] = provenance
    except Exception as exc:
        blockers.append(classify_blocker("setup", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    try:
        model = load_model_from_args(args, config)
        model.eval()
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    metadata = build_metadata_record(args, tokenizer=tokenizer, config=config, model=model, blockers=blockers, run_dir=run_dir)
    write_json(run_dir / "metadata.json", metadata)

    try:
        hidden_size = int(get_config_value(config, "hidden_size"))
        targets = torch.tensor([row.target for row in rows], dtype=torch.float32)
        if int(targets.shape[-1]) != hidden_size:
            raise ValueError(f"activation width={int(targets.shape[-1])}, expected hidden_size={hidden_size}")
        train_indices, eval_indices = split_indices(
            len(rows),
            args.train_fraction,
            strategy=args.split_strategy,
            seed=args.random_seed,
            records=[{"doc_id": row.metadata.get("doc_id")} for row in rows],
        )
        train_targets = take_rows(targets, train_indices)
        payload["split_metadata"] = split_metadata(
            [{"doc_id": row.metadata.get("doc_id")} for row in rows],
            train_indices,
            eval_indices,
        )
        payload["examples"] = [
            {
                "record_id": row.record_id,
                "row_index": row.row_index,
                "boundary_b": row.boundary_b,
                "doc_id": row.metadata.get("doc_id"),
                "target_l2": float(torch.tensor(row.target).norm().item()),
                "has_api_explanation": row.api_explanation is not None,
                "has_source_text": row.source_text is not None,
                "has_source_token_ids": row.source_token_ids is not None,
            }
            for row in rows[: min(8, len(rows))]
        ]

        variant_results: dict[str, dict[str, Any]] = {}
        for variant in args.variants:
            items = build_variant_items(
                rows,
                variant=variant,
                critic_template=args.critic_template,
                generic_explanation=args.generic_explanation,
                seed=args.random_seed + 17,
            )
            features, feature_records = materialize_variant_features(
                model=model,
                tokenizer=tokenizer,
                items=items,
                boundary_b=boundary_b,
                max_length=args.prompt_max_length,
                batch_size=args.feature_batch_size,
                tau=args.tau,
            )
            result = variant_eval(
                features=features,
                targets=targets,
                train_indices=train_indices,
                eval_indices=eval_indices,
                max_steps=args.max_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
                log_every=args.log_every,
                train_targets_for_mean=train_targets,
            )
            tracker.log_history(result.get("history"), prefix=f"{variant}/train")
            result["feature_records_sample"] = feature_records[: min(8, len(feature_records))]
            result["provenance_counts"] = {
                name: sum(1 for record in feature_records if record["provenance"] == name)
                for name in sorted({record["provenance"] for record in feature_records})
            }
            variant_results[variant] = result

        payload["variant_results"] = variant_results
        payload["comparison"] = compare_teacher_controls(
            variant_results,
            mse_margin=args.mse_margin,
            cosine_margin=args.cosine_margin,
            oracle_mse_threshold=args.oracle_mse_threshold,
        )
        payload["scientific_passed"] = bool(
            payload["comparison"].get("teacher_beats_controls")
            and payload["comparison"].get("source_raw_oracle_passed")
        )
        payload["passed"] = not blockers and bool(variant_results)
        if payload["data_source"]["provenance"]["source_token_ids_count"] == 0:
            payload["warnings"] = [
                {
                    "kind": "provenance",
                    "label": "source token ids",
                    "message": "source_raw used re-tokenized source text because token_ids_prefix is absent; use exact-provenance rows before scaled runs",
                }
            ]
    except Exception as exc:
        blockers.append(classify_blocker("signal gate", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))

    payload["blockers"] = blockers
    if blockers:
        payload["passed"] = False
    tracker.log_summary(payload)
    tracker.finish({"status/passed": bool(payload["passed"]), "status/blockers": len(payload["blockers"])})
    write_json(output_path, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
