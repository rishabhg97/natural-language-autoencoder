#!/usr/bin/env python3
"""Nano model loading and batched boundary-patching runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nano_eval_core import functional_logit_metrics
from nano_functional_eval_data import FunctionalEvaluationError
from nano_r33_source_rows import provenance_key


def _torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise FunctionalEvaluationError(f"unsupported torch dtype: {name}")
    return aliases[name]


def prepare_local_target_remote_code(
    target_model: str | Path,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> bool:
    """Apply the canonical Nemotron-H compatibility patch to local snapshots."""

    checkpoint = Path(target_model)
    if not trust_remote_code or not local_files_only or not checkpoint.is_dir():
        return False
    from nla.remote_code_patches import prepare_nemotron_h_checkpoint_for_load

    report = prepare_nemotron_h_checkpoint_for_load(checkpoint)
    return report is not None


def load_target_model(args: Any):
    import torch
    from transformers import AutoModelForCausalLM

    prepare_local_target_remote_code(
        args.target_model,
        trust_remote_code=bool(args.target_trust_remote_code),
        local_files_only=bool(args.target_local_files_only),
    )

    kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(args.target_torch_dtype),
        "trust_remote_code": args.target_trust_remote_code,
        "local_files_only": args.target_local_files_only,
    }
    if args.target_revision:
        kwargs["revision"] = args.target_revision
    if args.target_device_map != "none":
        kwargs["device_map"] = args.target_device_map
    model = AutoModelForCausalLM.from_pretrained(args.target_model, **kwargs).eval()
    if args.target_device_map == "none" and torch.cuda.is_available():
        model.cuda()
    return model


def _module_device(module: Any):
    import torch

    for parameter in module.parameters(recurse=True):
        if parameter.device.type != "meta":
            return parameter.device
    for buffer in module.buffers(recurse=True):
        if buffer.device.type != "meta":
            return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_logits(output: Any):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise FunctionalEvaluationError("target model output did not contain logits")


def _pad_prefixes(prefixes: list[list[int]], *, pad_token_id: int, device: Any):
    import torch

    if not prefixes or any(not prefix for prefix in prefixes):
        raise FunctionalEvaluationError("every source row needs a non-empty token_ids_prefix")
    max_length = max(len(prefix) for prefix in prefixes)
    input_ids = torch.full(
        (len(prefixes), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = torch.empty((len(prefixes),), dtype=torch.long, device=device)
    for row_index, prefix in enumerate(prefixes):
        values = torch.tensor(prefix, dtype=torch.long, device=device)
        input_ids[row_index, : len(prefix)] = values
        attention_mask[row_index, : len(prefix)] = 1
        positions[row_index] = len(prefix) - 1
    return input_ids, attention_mask, positions


def _capture_boundary_forward(
    model: Any,
    boundary_module: Any,
    input_ids: Any,
    attention_mask: Any,
    positions: Any,
):
    import torch

    from nano_r33_functional_core import gather_position_logits

    captured: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().clone()

    handle = boundary_module.register_forward_hook(capture)
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise FunctionalEvaluationError("R33 boundary capture hook did not fire")
    return (
        gather_position_logits(captured["hidden"], positions),
        gather_position_logits(_model_logits(output), positions),
    )


def _patched_forward(
    model: Any,
    boundary_module: Any,
    input_ids: Any,
    attention_mask: Any,
    positions: Any,
    replacements: Any,
):
    import torch

    from nano_r33_functional_core import gather_position_logits, make_boundary_replacement_hook

    handle = boundary_module.register_forward_hook(
        make_boundary_replacement_hook(replacements, positions=positions)
    )
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    return gather_position_logits(_model_logits(output), positions)


def run_identity_pass(
    *,
    model: Any,
    boundary_module: Any,
    selected: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], Any]]:
    import torch

    from nano_extraction_identity import tensor_metrics

    start_device = _module_device(model.get_input_embeddings())
    identity_rows: list[dict[str, Any]] = []
    original_logits: dict[tuple[Any, ...], Any] = {}
    for start in range(0, len(selected), batch_size):
        records_chunk = selected[start : start + batch_size]
        sources_chunk = sources[start : start + batch_size]
        input_ids, attention_mask, positions = _pad_prefixes(
            [[int(token) for token in row["token_ids_prefix"]] for row in sources_chunk],
            pad_token_id=pad_token_id,
            device=start_device,
        )
        boundary_values, baseline_logits = _capture_boundary_forward(
            model,
            boundary_module,
            input_ids,
            attention_mask,
            positions,
        )
        identity_logits = _patched_forward(
            model,
            boundary_module,
            input_ids,
            attention_mask,
            positions,
            boundary_values,
        )
        for offset, (record, source) in enumerate(
            zip(records_chunk, sources_chunk, strict=True)
        ):
            key = provenance_key(record)
            gold = torch.tensor(
                source["activation_vector"],
                dtype=torch.float32,
                device=boundary_values.device,
            )
            identity_rows.append(
                {
                    "split": str(record["split"]),
                    "row_index": int(record["row_index"]),
                    "provenance_key": list(key),
                    "stored_activation_drift": tensor_metrics(
                        boundary_values[offset], gold
                    ),
                    "logit_identity": tensor_metrics(
                        identity_logits[offset], baseline_logits[offset]
                    ),
                }
            )
            original_logits[key] = baseline_logits[offset].detach().float().cpu()
    return identity_rows, original_logits


def run_functional_pass(
    *,
    model: Any,
    boundary_module: Any,
    entries: list[dict[str, Any]],
    original_logits: dict[tuple[Any, ...], Any],
    batch_size: int,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    import torch

    start_device = _module_device(model.get_input_embeddings())
    output: list[dict[str, Any]] = []
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        input_ids, attention_mask, positions = _pad_prefixes(
            [entry["prefix"] for entry in chunk],
            pad_token_id=pad_token_id,
            device=start_device,
        )
        replacements = torch.stack([entry["replacement"] for entry in chunk])
        patched = _patched_forward(
            model,
            boundary_module,
            input_ids,
            attention_mask,
            positions,
            replacements,
        ).detach().float().cpu()
        for offset, entry in enumerate(chunk):
            key = tuple(entry["provenance_key"])
            output.append(
                {
                    "split": entry["split"],
                    "row_index": int(entry["row_index"]),
                    "provenance_key": list(key),
                    "content_family_id": entry.get("content_family_id"),
                    "variant": entry["variant"],
                    "metrics": functional_logit_metrics(
                        original_logits[key].numpy(), patched[offset].numpy()
                    ),
                }
            )
    return output
