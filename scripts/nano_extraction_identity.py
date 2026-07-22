#!/usr/bin/env python3
"""Nano30B residual-boundary extraction identity harness.

Checks, for each requested prompt mode and boundary:

    hooked R_b ~= output_hidden_states[b] ~= prefix_forward_to_R_b

The default boundary order is R_34 first, then R_27. The script writes JSON
after each check and exits nonzero on the first failure unless
`--continue-on-failure` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple

try:
    import torch
except ModuleNotFoundError:
    torch = None


def no_grad_context(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
    render_chat_prompt,
    resolve_nano_module_paths,
    write_json,
)


RAW_PROMPT = (
    "The Nano NLA pilot checks whether a residual-stream boundary can be extracted "
    "reproducibly before any autoencoder training starts."
)

CHAT_PROMPT = (
    "In one concise sentence, describe why residual-boundary identity checks should "
    "run before training."
)

AV_MARKER_PROMPT = """You are a meticulous AI researcher investigating activation vectors from a language model.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the
vector, enclosed within <explanation> tags.

Here is the vector:

<concept><NLA_ACTIVATION_MARKER></concept>

Please provide an explanation."""

AR_CRITIC_PROMPT = (
    "Summary of the following text: <text>The activation appears to encode a boundary-test "
    "diagnostic rather than a source-context summary.</text> <summary>"
)


class IdentityTolerances(NamedTuple):
    relative_l2: float = 1e-2
    max_abs: float = 1e-2
    one_minus_cos: float = 1e-4


class PromptInputs(NamedTuple):
    name: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: dict[str, Any]


def _as_output_hidden_states(output: Any) -> tuple[torch.Tensor, ...]:
    if hasattr(output, "hidden_states"):
        return tuple(output.hidden_states)
    if isinstance(output, dict) and "hidden_states" in output:
        return tuple(output["hidden_states"])
    if isinstance(output, (tuple, list)) and output:
        candidate = output[-1]
        if isinstance(candidate, (tuple, list)):
            return tuple(candidate)
    raise RuntimeError("model output did not expose hidden_states")


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _move_inputs_to_model_start(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = _model_start_device(model)
    if device.type == "meta":
        raise RuntimeError("identity checks require real weights; model is on meta device")
    return input_ids.to(device), attention_mask.to(device)


def _module_execution_device(module: Any, fallback: torch.device) -> torch.device:
    for parameter in module.parameters(recurse=True):
        if parameter.device.type != "meta":
            return parameter.device
    for buffer in module.buffers(recurse=True):
        if buffer.device.type != "meta":
            return buffer.device
    return fallback


def _move_optional_tensor(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if tensor is None or tensor.device == device:
        return tensor
    return tensor.to(device)


def _layer_mask_for_block(backbone: Any, block: Any, attention_mask: torch.Tensor | None, hidden_states: torch.Tensor, cache_position: torch.Tensor):
    attention_mask = _move_optional_tensor(attention_mask, hidden_states.device)
    if cache_position.device != hidden_states.device:
        cache_position = cache_position.to(hidden_states.device)
    if hasattr(backbone, "_update_causal_mask"):
        causal_mask = backbone._update_causal_mask(attention_mask, hidden_states, cache_position)
    else:
        causal_mask = attention_mask
    if hasattr(backbone, "_update_mamba_mask"):
        mamba_mask = backbone._update_mamba_mask(attention_mask, cache_position)
    else:
        mamba_mask = attention_mask

    block_type = str(getattr(block, "block_type", "")).lower()
    if block_type == "mamba":
        return mamba_mask
    if block_type == "attention":
        return causal_mask
    if block_type in {"mlp", "moe"}:
        return None

    class_hint = f"{type(block).__name__} {type(getattr(block, 'mixer', None)).__name__}".lower()
    if "mamba" in class_hint:
        return mamba_mask
    if "attention" in class_hint:
        return causal_mask
    return None


@no_grad_context
def prefix_forward_to_R_b(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    boundary_b: int,
) -> torch.Tensor:
    """Run embeddings plus blocks 0..b-1 and return raw residual boundary R_b."""
    resolved = resolve_nano_module_paths(model)
    backbone = resolved["backbone"].obj
    layers = resolved["layers"].obj
    embeddings = resolved["embeddings"].obj
    if backbone is None or layers is None or embeddings is None:
        raise RuntimeError(f"could not resolve Nano modules: {json_safe(resolved)}")
    if not 0 <= boundary_b <= len(layers):
        raise ValueError(f"boundary_b={boundary_b} out of range for {len(layers)} blocks")

    hidden_states = embeddings(input_ids)
    if boundary_b == 0:
        return hidden_states

    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    for layer_idx in range(boundary_b):
        block = layers[layer_idx]
        block_device = _module_execution_device(block, hidden_states.device)
        if hidden_states.device != block_device:
            hidden_states = hidden_states.to(block_device)
        block_attention_mask = _move_optional_tensor(attention_mask, block_device)
        block_cache_position = cache_position.to(block_device) if cache_position.device != block_device else cache_position
        layer_mask = _layer_mask_for_block(backbone, block, block_attention_mask, hidden_states, block_cache_position)
        output = block(
            hidden_states,
            cache_params=None,
            cache_position=block_cache_position,
            attention_mask=layer_mask,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
    return hidden_states


def tensor_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float | list[int] | str]:
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    diff = lhs_f - rhs_f
    lhs_flat = lhs_f.reshape(-1)
    rhs_flat = rhs_f.reshape(-1)
    diff_flat = diff.reshape(-1)
    rhs_norm = rhs_flat.norm().clamp_min(1e-12)
    lhs_norm = lhs_flat.norm().clamp_min(1e-12)
    cosine = torch.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm)
    lhs_rms = lhs_flat.pow(2).mean().sqrt().clamp_min(1e-12)
    rhs_rms = rhs_flat.pow(2).mean().sqrt().clamp_min(1e-12)
    return {
        "shape": [int(x) for x in lhs.shape],
        "lhs_dtype": str(lhs.dtype),
        "rhs_dtype": str(rhs.dtype),
        "relative_l2": float((diff_flat.norm() / rhs_norm).item()),
        "max_abs": float(diff_flat.abs().max().item()),
        "one_minus_cos": float((1.0 - cosine).item()),
        "rms_ratio": float((lhs_rms / rhs_rms).item()),
    }


def metrics_pass(metrics: dict[str, Any], tolerances: IdentityTolerances) -> bool:
    return (
        metrics["relative_l2"] <= tolerances.relative_l2
        and metrics["max_abs"] <= tolerances.max_abs
        and metrics["one_minus_cos"] <= tolerances.one_minus_cos
    )


def _capture_hook_output(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    return tensor.detach().clone()


@no_grad_context
def run_boundary_identity_check(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_b: int,
    prompt_name: str,
    tolerances: IdentityTolerances,
) -> dict[str, Any]:
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    embeddings = resolved["embeddings"].obj
    if layers is None or embeddings is None:
        raise RuntimeError(f"could not resolve layers/embeddings: {json_safe(resolved)}")
    if not 0 <= boundary_b < len(layers):
        raise ValueError(f"boundary_b={boundary_b} requires 0 <= b < {len(layers)} for hook/output_hidden_states checks")

    input_ids, attention_mask = _move_inputs_to_model_start(model, input_ids, attention_mask)
    captured: dict[str, torch.Tensor] = {}
    if boundary_b == 0:
        hook_path = resolved["embeddings"].path
        captured["hooked"] = embeddings(input_ids).detach().clone()
        handle = None
    else:
        hook_path = f"{resolved['layers'].path}.{boundary_b - 1}"
        handle = layers[boundary_b - 1].register_forward_hook(
            lambda _module, _inputs, output: captured.setdefault("hooked", _capture_hook_output(output))
        )

    try:
        full = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    finally:
        if handle is not None:
            handle.remove()

    states = _as_output_hidden_states(full)
    if boundary_b >= len(states):
        raise RuntimeError(f"output_hidden_states has {len(states)} entries; cannot read index {boundary_b}")
    if "hooked" not in captured:
        raise RuntimeError(f"forward hook did not fire for {hook_path}")

    hooked = captured["hooked"]
    output_hidden = states[boundary_b].detach()
    prefix = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b).detach()

    comparisons = {
        "hook_vs_output_hidden_states": tensor_metrics(hooked, output_hidden),
        "hook_vs_prefix_forward": tensor_metrics(hooked, prefix),
        "output_hidden_states_vs_prefix_forward": tensor_metrics(output_hidden, prefix),
    }
    comparison_pass = {name: metrics_pass(value, tolerances) for name, value in comparisons.items()}
    passed = all(comparison_pass.values())

    return {
        "prompt_name": prompt_name,
        "boundary_b": boundary_b,
        "zero_based_block_index": boundary_b - 1,
        "hook_path": hook_path,
        "output_hidden_states_index": boundary_b,
        "final_norm_removed_or_bypassed": True,
        "comparisons": comparisons,
        "comparison_pass": comparison_pass,
        "passed": passed,
    }


def _truncate(ids: list[int], max_length: int | None) -> list[int]:
    if max_length is not None and len(ids) > max_length:
        return ids[:max_length]
    return ids


def _tensorize_ids(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def encode_raw_prompt(tokenizer: Any, name: str, text: str, max_length: int | None) -> PromptInputs:
    kwargs: dict[str, Any] = {"return_tensors": "pt", "add_special_tokens": True}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    enc = tokenizer(text, **kwargs)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    return PromptInputs(
        name=name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        metadata={
            "prompt_format": "raw",
            "token_count": int(input_ids.shape[-1]),
            "rendered_sha256": None,
            "enable_thinking": None,
        },
    )


def encode_chat_prompt(
    tokenizer: Any,
    name: str,
    content: str,
    max_length: int | None,
    *,
    add_generation_prompt: bool = True,
) -> PromptInputs:
    rendered = render_chat_prompt(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    ids = _truncate(rendered.token_ids, max_length)
    input_ids, attention_mask = _tensorize_ids(ids)
    return PromptInputs(
        name=name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        metadata={
            "prompt_format": "chat",
            "token_count": len(ids),
            "rendered_sha256": rendered.sha256,
            "add_generation_prompt": rendered.add_generation_prompt,
            "enable_thinking_requested": rendered.enable_thinking_requested,
            "enable_thinking_applied": rendered.enable_thinking_applied,
            "template_error": rendered.template_error,
        },
    )


def build_prompt_inputs(tokenizer: Any, max_length: int | None) -> list[PromptInputs]:
    return [
        encode_raw_prompt(tokenizer, "raw", RAW_PROMPT, max_length),
        encode_chat_prompt(tokenizer, "reasoning_off_chat", CHAT_PROMPT, max_length),
        encode_chat_prompt(tokenizer, "av_marker", AV_MARKER_PROMPT, max_length),
        encode_raw_prompt(tokenizer, "ar_critic", AR_CRITIC_PROMPT, max_length),
    ]


def parse_boundaries(value: str) -> list[int]:
    boundaries = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item.startswith("R_"):
            item = item[2:]
        boundaries.append(int(item))
    if not boundaries:
        raise argparse.ArgumentTypeError("at least one boundary is required")
    return boundaries


def classify_identity_failure(check: dict[str, Any]) -> list[dict[str, str]]:
    blockers = []
    passes = check.get("comparison_pass", {})
    if passes.get("hook_vs_output_hidden_states") is False:
        blockers.append(
            {
                "kind": "output_hidden_states mismatch",
                "label": f"{check['prompt_name']} R_{check['boundary_b']}",
                "error": json.dumps(check["comparisons"]["hook_vs_output_hidden_states"], sort_keys=True),
            }
        )
    if passes.get("hook_vs_prefix_forward") is False or passes.get("output_hidden_states_vs_prefix_forward") is False:
        blockers.append(
            {
                "kind": "final-norm mismatch",
                "label": f"{check['prompt_name']} R_{check['boundary_b']}",
                "error": "prefix_forward_to_R_b did not match hooked/output_hidden_states boundary; check off-by-one, masks, or final norm",
            }
        )
    return blockers


def identity_payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_extraction_identity.v1",
        "run_dir": str(run_dir),
        "boundary_order": args.boundaries,
        "tolerances": {
            "relative_l2": args.relative_l2_tol,
            "max_abs": args.max_abs_tol,
            "one_minus_cos": args.one_minus_cos_tol,
        },
        "prompt_modes": [],
        "checks": [],
        "passed": False,
        "blockers": [],
    }


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
    parser.add_argument("--prompt-max-length", type=int, default=256)
    parser.add_argument("--relative-l2-tol", type=float, default=1e-2)
    parser.add_argument("--max-abs-tol", type=float, default=1e-2)
    parser.add_argument("--one-minus-cos-tol", type=float, default=1e-4)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    identity = identity_payload_base(args, run_dir)
    identity_path = run_dir / "identity.json"
    if torch is None:
        identity["blockers"] = [
            {
                "kind": "environment",
                "label": "torch import",
                "error": "PyTorch is required for extraction identity checks but is not installed in this Python environment.",
            }
        ]
        write_json(identity_path, identity)
        print(json.dumps(json_safe(identity), indent=2, sort_keys=True))
        print(f"\nwrote {identity_path}")
        return 2

    blockers: list[dict[str, str]] = []
    tokenizer = None
    config = None
    model = None
    prompts: list[PromptInputs] = []
    try:
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
    except Exception as exc:
        blockers.append(classify_blocker("tokenizer/config load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        identity["blockers"] = blockers
        write_json(identity_path, identity)
        print(json.dumps(json_safe(identity), indent=2, sort_keys=True))
        print(f"\nwrote {identity_path}")
        return 2

    try:
        prompts = build_prompt_inputs(tokenizer, args.prompt_max_length)
        identity["prompt_modes"] = [{"name": item.name, **item.metadata} for item in prompts]
    except Exception as exc:
        identity["blockers"].append(classify_blocker("template ambiguity", f"{type(exc).__name__}: {exc}"))
        write_json(identity_path, identity)
        print(json.dumps(json_safe(identity), indent=2, sort_keys=True))
        print(f"\nwrote {identity_path}")
        return 2

    try:
        model = load_model_from_args(args, config)
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        identity["blockers"] = blockers
        write_json(identity_path, identity)
        print(json.dumps(json_safe(identity), indent=2, sort_keys=True))
        print(f"\nwrote {identity_path}")
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

    pattern = block_pattern_from_config(config, resolve_nano_module_paths(model)["layers"].obj)
    identity["model"] = {
        "model_id": args.model_id,
        "revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        "hidden_size": get_config_value(config, "hidden_size"),
        "block_count": get_config_value(config, "num_hidden_layers"),
        "block_pattern": pattern,
    }

    tolerances = IdentityTolerances(
        relative_l2=args.relative_l2_tol,
        max_abs=args.max_abs_tol,
        one_minus_cos=args.one_minus_cos_tol,
    )

    model.eval()
    for boundary_b in args.boundaries:
        for prompt in prompts:
            try:
                check = run_boundary_identity_check(
                    model=model,
                    input_ids=prompt.input_ids,
                    attention_mask=prompt.attention_mask,
                    boundary_b=boundary_b,
                    prompt_name=prompt.name,
                    tolerances=tolerances,
                )
            except Exception as exc:
                check = {
                    "prompt_name": prompt.name,
                    "boundary_b": boundary_b,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
                }
                identity["blockers"].append(classify_blocker("boundary extraction", check["error"]))

            identity["checks"].append(check)
            if not check.get("passed", False):
                identity["blockers"].extend(classify_identity_failure(check))
                write_json(identity_path, identity)
                print(json.dumps(json_safe(identity), indent=2, sort_keys=True))
                print(f"\nwrote {identity_path}")
                if not args.continue_on_failure:
                    return 1
            else:
                write_json(identity_path, identity)

    identity["passed"] = all(check.get("passed", False) for check in identity["checks"])
    write_json(identity_path, identity)
    print(json.dumps(json_safe(identity), indent=2, sort_keys=True))
    print(f"\nwrote {identity_path}")
    return 0 if identity["passed"] and not identity["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
