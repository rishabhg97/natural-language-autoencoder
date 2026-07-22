#!/usr/bin/env python3
"""Nano30B Track C HF residual-boundary oracle probe.

Track C is a diagnostic, not a serving path:

    run AV prompt u to R_b
    patch sentinel position p: R_b[p] <- T(h_b)
    continue layers b..L-1 in the HF graph

The probe keeps Nano frozen, checks that split-forward without a patch matches
full forward, checks self-replacement identity, then measures next-token logit
effects for correct, shuffled, and random matched-norm residual patches. It does
not train, serve, generate datasets, run PEFT, or run RL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import MethodType
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_extraction_identity import (  # noqa: E402
    _layer_mask_for_block,
    build_prompt_inputs,
    parse_boundaries,
    prefix_forward_to_R_b,
)
from nano_extraction_serialize_probe import parse_prompt_names  # noqa: E402
from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_ROOT,
    add_bool_optional_arg,
    block_pattern_from_config,
    build_metadata_record,
    classify_blocker,
    get_config_value,
    json_safe,
    load_config_from_args,
    load_model_from_args,
    load_tokenizer_from_args,
    make_run_dir,
    resolve_nano_module_paths,
    write_json,
)
from nano_track_a_probe import (  # noqa: E402
    DEFAULT_CONTROLS,
    DEFAULT_INJECTION_TEXT,
    build_injection_prompt,
    control_vector_for,
    next_token_logit_metrics,
    normalize_activation,
    parse_controls,
    parse_float_list,
    source_vectors_from_prompts,
    summarize_alpha_response,
)


def _module_device(module: Any, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except Exception:
        try:
            return next(module.buffers()).device
        except Exception:
            return fallback


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _move_prompt_to_model(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = _model_start_device(model)
    return input_ids.to(device), attention_mask.to(device)


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise RuntimeError("model output did not expose logits")


def _lm_head(model: Any) -> Any:
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    raise RuntimeError("could not resolve LM head for suffix logits")


def _apply_lm_head(model: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    head = _lm_head(model)
    device = _module_device(head, hidden_states.device)
    hidden_states = hidden_states.to(device)
    weight = getattr(head, "weight", None)
    if weight is not None:
        hidden_states = hidden_states.to(dtype=weight.dtype)
    return head(hidden_states).float()


def replace_residual_row(residual: torch.Tensor, *, position: int, vector: torch.Tensor) -> torch.Tensor:
    """Clone a residual tensor and replace one sequence row with vector."""
    if residual.ndim != 3:
        raise ValueError(f"residual must be [B,T,d], got shape {tuple(residual.shape)}")
    if residual.shape[0] != 1:
        raise ValueError(f"Track C probe expects batch size 1, got {residual.shape[0]}")
    if not 0 <= position < residual.shape[1]:
        raise ValueError(f"position={position} out of range for sequence length {residual.shape[1]}")
    if vector.numel() != residual.shape[-1]:
        raise ValueError(f"vector has {vector.numel()} values but residual dim is {residual.shape[-1]}")
    out = residual.clone()
    out[0, position] = vector.reshape(-1).to(device=out.device, dtype=out.dtype)
    return out


def forward_full_next_logits(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
    return _extract_logits(output)[:, -1, :].detach()


def forward_suffix_from_R_b(
    model: Any,
    residual_b: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    boundary_b: int,
    cache_params: Any | None = None,
) -> torch.Tensor:
    """Continue raw residual boundary R_b through layers b..L-1 and lm_head."""
    resolved = resolve_nano_module_paths(model)
    backbone = resolved["backbone"].obj
    layers = resolved["layers"].obj
    norm_f = resolved["norm_f"].obj
    if backbone is None or layers is None or norm_f is None:
        raise RuntimeError(f"could not resolve Nano suffix modules: {json_safe(resolved)}")
    if not 0 <= boundary_b <= len(layers):
        raise ValueError(f"boundary_b={boundary_b} out of range for {len(layers)} blocks")

    hidden_states = residual_b
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    for layer_idx in range(boundary_b, len(layers)):
        block = layers[layer_idx]
        device = _module_device(block, hidden_states.device)
        hidden_states = hidden_states.to(device)
        layer_attention_mask = attention_mask.to(device) if attention_mask is not None else None
        layer_cache_position = cache_position.to(device)
        layer_mask = _layer_mask_for_block(backbone, block, layer_attention_mask, hidden_states, layer_cache_position)
        output = block(
            hidden_states,
            cache_params=cache_params,
            cache_position=layer_cache_position,
            attention_mask=layer_mask,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output

    hidden_states = hidden_states.to(_module_device(norm_f, hidden_states.device))
    hidden_states = norm_f(hidden_states)
    logits = _apply_lm_head(model, hidden_states)
    return logits[:, -1, :].detach()


def split_forward_next_logits(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_b: int,
    patch_position: int | None = None,
    patch_vector: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b)
    if patch_position is not None:
        if patch_vector is None:
            raise ValueError("patch_vector is required when patch_position is supplied")
        residual = replace_residual_row(residual, position=patch_position, vector=patch_vector)
    return forward_suffix_from_R_b(
        model,
        residual,
        attention_mask=attention_mask,
        boundary_b=boundary_b,
    )


def run_boundary_path_checks(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_b: int,
    patch_position: int,
    max_kl: float,
    max_abs: float,
) -> dict[str, Any]:
    full_logits = forward_full_next_logits(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    residual = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b)
    split_logits = forward_suffix_from_R_b(
        model,
        residual,
        attention_mask=attention_mask,
        boundary_b=boundary_b,
    )
    self_residual = replace_residual_row(
        residual,
        position=patch_position,
        vector=residual[0, patch_position],
    )
    self_logits = forward_suffix_from_R_b(
        model,
        self_residual,
        attention_mask=attention_mask,
        boundary_b=boundary_b,
    )
    split_metrics = next_token_logit_metrics(full_logits, split_logits)
    self_metrics = next_token_logit_metrics(full_logits, self_logits)
    split_passed = split_metrics["kl_baseline_to_variant"] <= max_kl and split_metrics["max_abs_logit_delta"] <= max_abs
    self_passed = self_metrics["kl_baseline_to_variant"] <= max_kl and self_metrics["max_abs_logit_delta"] <= max_abs
    return {
        "boundary_b": boundary_b,
        "split_no_patch_vs_full": split_metrics,
        "self_replacement_vs_full": self_metrics,
        "split_no_patch_passed": split_passed,
        "self_replacement_passed": self_passed,
        "passed": split_passed and self_passed,
    }


def _cache_class(model: Any) -> Any | None:
    module = sys.modules.get(type(model).__module__)
    return getattr(module, "HybridMambaAttentionDynamicCache", None) if module is not None else None


def _cache_state_device_dtype(state: Any, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype | None]:
    device = getattr(state, "device", fallback.device)
    dtype = getattr(state, "dtype", None)
    return device, dtype


def _to_cache_state(tensor: torch.Tensor, state: Any) -> torch.Tensor:
    device, dtype = _cache_state_device_dtype(state, tensor)
    if dtype is None:
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def make_hybrid_cache_compatible(cache: Any, config: Any) -> list[str]:
    """Repair pinned Nano remote-code cache fields used by Mamba prefill."""
    repairs: list[str] = []
    if not hasattr(cache, "conv_kernel_size"):
        conv_kernel_size = get_config_value(config, "conv_kernel", None)
        if conv_kernel_size is None and getattr(cache, "conv_states", None):
            conv_kernel_size = cache.conv_states[0].shape[-1]
        if conv_kernel_size is None:
            raise RuntimeError("could not infer conv_kernel_size for HybridMambaAttentionDynamicCache")
        cache.conv_kernel_size = int(conv_kernel_size)
        repairs.append("set conv_kernel_size")

    def update_conv_state(self: Any, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False) -> torch.Tensor:
        current = self.conv_states[layer_idx]
        if cache_init:
            self.conv_states[layer_idx] = _to_cache_state(new_conv_state, current)
        else:
            updated = current.roll(shifts=-1, dims=-1)
            updated[:, :, -1] = _to_cache_state(new_conv_state[:, 0, :], updated)
            self.conv_states[layer_idx] = updated
        return self.conv_states[layer_idx]

    def update_ssm_state(self: Any, layer_idx: int, new_ssm_state: torch.Tensor) -> torch.Tensor:
        current = self.ssm_states[layer_idx]
        self.ssm_states[layer_idx] = _to_cache_state(new_ssm_state, current)
        return self.ssm_states[layer_idx]

    def reset(self: Any) -> None:
        for state in getattr(self, "conv_states", []):
            if hasattr(state, "zero_"):
                state.zero_()
        for state in getattr(self, "ssm_states", []):
            if hasattr(state, "zero_"):
                state.zero_()

    cache.update_conv_state = MethodType(update_conv_state, cache)
    cache.update_ssm_state = MethodType(update_ssm_state, cache)
    cache.reset = MethodType(reset, cache)
    repairs.extend(["patched update_conv_state", "patched update_ssm_state", "patched reset"])
    return repairs


def run_cache_api_smoke(
    *,
    model: Any,
    config: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_b: int,
) -> dict[str, Any]:
    """Best-effort cache API smoke; failure is diagnostic, not a serving path."""
    cache_cls = _cache_class(model)
    if cache_cls is None:
        return {"attempted": True, "passed": False, "error": "HybridMambaAttentionDynamicCache not found in model module"}
    try:
        dtype = next(model.parameters()).dtype
        device = _model_start_device(model)
        cache = cache_cls(config, batch_size=int(input_ids.shape[0]), dtype=dtype, device=device)
        repairs = make_hybrid_cache_compatible(cache, config)
        residual = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b)
        _ = forward_suffix_from_R_b(
            model,
            residual,
            attention_mask=attention_mask,
            boundary_b=boundary_b,
            cache_params=cache,
        )
        return {
            "attempted": True,
            "passed": True,
            "class": f"{cache_cls.__module__}.{cache_cls.__name__}",
            "compatibility_repairs": repairs,
            "note": "suffix prefill cache update ran without exception",
        }
    except Exception as exc:
        return {
            "attempted": True,
            "passed": False,
            "class": f"{cache_cls.__module__}.{cache_cls.__name__}",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
        }


def control_gap_passed(summary: dict[str, Any]) -> bool:
    shuffled_gap = summary.get("correct_vs_shuffled_kl_gap_at_max_alpha")
    random_gap = summary.get("correct_vs_random_kl_gap_at_max_alpha")
    return bool(shuffled_gap is not None and shuffled_gap > 0 and random_gap is not None and random_gap > 0)


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
    parser.add_argument("--boundaries", type=parse_boundaries, default=[34, 27])
    parser.add_argument("--prompt-names", type=parse_prompt_names, default=["raw", "reasoning_off_chat", "av_marker", "ar_critic"])
    parser.add_argument("--prompt-max-length", type=int, default=256)
    parser.add_argument("--source-tau", type=int, default=-1)
    parser.add_argument("--injection-text", default=DEFAULT_INJECTION_TEXT)
    parser.add_argument("--marker-token-strategy", choices=("first", "middle", "last"), default="first")
    parser.add_argument("--alphas", type=parse_float_list, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--controls", type=parse_controls, default=list(DEFAULT_CONTROLS))
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--path-equivalence-max-kl", type=float, default=1e-4)
    parser.add_argument("--path-equivalence-max-abs", type=float, default=1e-2)
    parser.add_argument("--cache-smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_track_c_probe.v1",
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "track": "C",
        "definition": "HF split-forward residual-boundary replacement",
        "serving_path": False,
        "boundary_order": args.boundaries,
        "source_tau": args.source_tau,
        "requested_prompt_names": args.prompt_names,
        "alphas": args.alphas,
        "controls": args.controls,
        "injection_prompt": {},
        "path_checks": [],
        "cache_smoke": {"attempted": False},
        "source_vectors": [],
        "records": [],
        "summary": {},
        "path_passed": False,
        "signal_non_flat": False,
        "control_gap_passed": False,
        "passed": False,
        "blockers": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    payload = payload_base(args, run_dir)
    output_path = run_dir / "track_c_probe.json"

    if torch is None:
        payload["blockers"] = [{"kind": "environment", "label": "torch import", "error": "PyTorch is required for Track C probe."}]
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    blockers: list[dict[str, str]] = []
    try:
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
        prompts = build_prompt_inputs(tokenizer, args.prompt_max_length)
        prompts = [item for item in prompts if item.name in args.prompt_names]
        prompt_by_name = {item.name: item for item in prompts}
        missing = sorted(set(args.prompt_names) - set(prompt_by_name))
        if missing:
            raise ValueError(f"requested prompt names were not built: {missing}")
        injection_prompt = build_injection_prompt(
            tokenizer,
            injection_text=args.injection_text,
            marker_token_strategy=args.marker_token_strategy,
        )
        payload["prompt_modes"] = [{"name": item.name, **item.metadata} for item in prompts]
        payload["injection_prompt"] = {
            "marker_text": args.injection_text,
            "marker_token_positions": injection_prompt["marker_token_positions"],
            "patch_position": injection_prompt["injection_position"],
            "marker_token_strategy": args.marker_token_strategy,
            "token_count": int(injection_prompt["input_ids"].shape[-1]),
            "rendered_sha256": injection_prompt["rendered"].sha256,
            "enable_thinking_requested": injection_prompt["rendered"].enable_thinking_requested,
            "enable_thinking_applied": injection_prompt["rendered"].enable_thinking_applied,
            "template_error": injection_prompt["rendered"].template_error,
        }
    except Exception as exc:
        blockers.append(classify_blocker("template ambiguity", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    try:
        model = load_model_from_args(args, config)
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    metadata = build_metadata_record(
        args,
        tokenizer=tokenizer,
        config=config,
        model=model,
        blockers=blockers,
        run_dir=run_dir,
    )
    write_json(run_dir / "metadata.json", metadata)
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    if layers is None:
        blockers.append(classify_blocker("model load", f"could not resolve layers: {json_safe(resolved)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    payload["model"].update(
        {
            "hidden_size": get_config_value(config, "hidden_size"),
            "block_count": get_config_value(config, "num_hidden_layers"),
            "block_pattern": block_pattern_from_config(config, layers),
        }
    )

    try:
        model.eval()
        source_vectors = source_vectors_from_prompts(
            model=model,
            prompts=prompts,
            boundaries=args.boundaries,
            tau=args.source_tau,
        )
        payload["source_vectors"] = [
            {key: value for key, value in item.items() if key != "vector"}
            for item in source_vectors
        ]

        input_ids, attention_mask = _move_prompt_to_model(
            model,
            injection_prompt["input_ids"],
            injection_prompt["attention_mask"],
        )
        patch_position = int(injection_prompt["injection_position"])
        boundary_residuals: dict[int, torch.Tensor] = {}
        boundary_full_logits: dict[int, torch.Tensor] = {}

        for boundary_b in args.boundaries:
            check = run_boundary_path_checks(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_b=boundary_b,
                patch_position=patch_position,
                max_kl=args.path_equivalence_max_kl,
                max_abs=args.path_equivalence_max_abs,
            )
            payload["path_checks"].append(check)
            boundary_residuals[boundary_b] = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b)
            boundary_full_logits[boundary_b] = forward_full_next_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            write_json(output_path, payload)

        if args.cache_smoke:
            payload["cache_smoke"] = run_cache_api_smoke(
                model=model,
                config=config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_b=args.boundaries[0],
            )

        for source_index, source in enumerate(source_vectors):
            boundary_b = int(source["boundary_b"])
            base_residual = boundary_residuals[boundary_b]
            baseline_logits = boundary_full_logits[boundary_b]
            for control in args.controls:
                control_vector, control_meta = control_vector_for(
                    source_index=source_index,
                    sources=source_vectors,
                    control=control,
                    seed=args.random_seed,
                )
                source_norm = float(source["vector"].float().norm().item())
                for alpha in args.alphas:
                    target_scale = float(alpha) * source_norm
                    scaled = normalize_activation(control_vector, target_scale=target_scale)
                    patched_residual = replace_residual_row(
                        base_residual,
                        position=patch_position,
                        vector=scaled,
                    )
                    variant_logits = forward_suffix_from_R_b(
                        model,
                        patched_residual,
                        attention_mask=attention_mask,
                        boundary_b=boundary_b,
                    )
                    metrics = next_token_logit_metrics(baseline_logits, variant_logits)
                    record = {
                        **metrics,
                        "boundary_b": boundary_b,
                        "source_prompt_name": source["prompt_name"],
                        "source_index": source_index,
                        "source_tau": source["tau"],
                        "resolved_tau": source["resolved_tau"],
                        "source_token_count": source["token_count"],
                        "control": control,
                        "control_source_index": control_meta["source_index"],
                        "alpha": float(alpha),
                        "target_scale": target_scale,
                        "patch_vector_l2": float(scaled.float().norm().item()),
                        "injection_mode": "residual_boundary_replacement",
                        "patch_position": patch_position,
                    }
                    payload["records"].append(record)
                    write_json(output_path, payload)

        summary = summarize_alpha_response(payload["records"])
        payload["summary"] = summary
        payload["signal_non_flat"] = bool(summary["correct_non_flat"])
        payload["control_gap_passed"] = control_gap_passed(summary)
        payload["path_passed"] = all(check.get("passed", False) for check in payload["path_checks"]) and not blockers
        payload["passed"] = payload["path_passed"] and payload["signal_non_flat"] and payload["control_gap_passed"]

        if not payload["path_passed"]:
            blockers.append(
                {
                    "kind": "Track C split-forward mismatch",
                    "label": "no-patch/self-replacement",
                    "error": json.dumps(json_safe(payload["path_checks"]), sort_keys=True),
                }
            )
    except Exception as exc:
        blockers.append(classify_blocker("Track C probe", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))

    payload["blockers"] = blockers
    if blockers:
        payload["passed"] = False
        payload["path_passed"] = False
    write_json(output_path, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
