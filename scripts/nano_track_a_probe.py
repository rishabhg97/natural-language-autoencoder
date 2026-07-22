#!/usr/bin/env python3
"""Nano30B Track A input-embedding injection probe.

Track A is the paper-faithful NLA channel:

    E(u)[p] <- alpha * h_b / ||h_b||

This probe keeps Nano frozen, extracts a few validated R_b vectors, injects
them into one AV-style prompt through `inputs_embeds`, and records downstream
next-token logit effects against shuffled/random controls. It does not train,
serve, generate datasets, run PEFT, or run RL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_extraction_identity import (  # noqa: E402
    AV_MARKER_PROMPT,
    build_prompt_inputs,
    parse_boundaries,
)
from nano_extraction_serialize_probe import (  # noqa: E402
    capture_boundary_tensor,
    parse_prompt_names,
)
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


DEFAULT_INJECTION_TEXT = "the"
DEFAULT_CONTROLS = ("correct", "shuffled", "random_matched_norm")


def normalize_activation(vector: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Rescale a vector to target L2 norm in fp32; zero vectors stay zero."""
    if target_scale < 0:
        raise ValueError(f"target_scale must be non-negative, got {target_scale}")
    vector_f = vector.float()
    norm = vector_f.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > 0, torch.as_tensor(target_scale, device=vector.device, dtype=vector_f.dtype) / norm.clamp_min(1e-12), torch.zeros_like(norm))
    return (vector_f * scale).to(vector.dtype)


def replace_embedding_row(embeddings: torch.Tensor, *, position: int, vector: torch.Tensor) -> torch.Tensor:
    """Clone embeddings and replace one sequence row with vector."""
    if embeddings.ndim != 3:
        raise ValueError(f"embeddings must be [B,T,d], got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != 1:
        raise ValueError(f"Track A probe expects batch size 1, got {embeddings.shape[0]}")
    if not 0 <= position < embeddings.shape[1]:
        raise ValueError(f"position={position} out of range for sequence length {embeddings.shape[1]}")
    if vector.numel() != embeddings.shape[-1]:
        raise ValueError(f"vector has {vector.numel()} values but embedding dim is {embeddings.shape[-1]}")
    out = embeddings.clone()
    out[0, position] = vector.reshape(-1).to(device=out.device, dtype=out.dtype)
    return out


def _offset_overlaps(offset: tuple[int, int], start: int, end: int) -> bool:
    left, right = offset
    return left < end and right > start


def marker_token_positions(
    *,
    rendered_text: str,
    offsets: list[tuple[int, int]],
    marker_text: str,
    left_context: str | None = None,
    right_context: str | None = None,
) -> list[int]:
    """Return token positions whose offsets overlap the marker substring."""
    if left_context is not None or right_context is not None:
        left = left_context or ""
        right = right_context or ""
        contextual_marker = f"{left}{marker_text}{right}"
        context_start = rendered_text.find(contextual_marker)
        if context_start < 0:
            raise ValueError(f"contextual marker not found in rendered prompt: {contextual_marker!r}")
        marker_start = context_start + len(left)
    else:
        marker_start = rendered_text.find(marker_text)
    if marker_start < 0:
        raise ValueError(f"marker text not found in rendered prompt: {marker_text!r}")
    marker_end = marker_start + len(marker_text)
    positions = [idx for idx, offset in enumerate(offsets) if _offset_overlaps(offset, marker_start, marker_end)]
    if not positions:
        raise ValueError(f"marker text {marker_text!r} was found but no tokenizer offsets overlapped it")
    return positions


def find_marker_token_position(
    *,
    rendered_text: str,
    offsets: list[tuple[int, int]],
    marker_text: str,
    left_context: str | None = None,
    right_context: str | None = None,
    strategy: str = "first",
) -> int:
    positions = marker_token_positions(
        rendered_text=rendered_text,
        offsets=offsets,
        marker_text=marker_text,
        left_context=left_context,
        right_context=right_context,
    )
    if strategy == "first":
        return positions[0]
    if strategy == "middle":
        return positions[len(positions) // 2]
    if strategy == "last":
        return positions[-1]
    raise ValueError(f"unknown marker token strategy {strategy!r}; expected first, middle, or last")


def next_token_logit_metrics(baseline_logits: torch.Tensor, variant_logits: torch.Tensor) -> dict[str, Any]:
    """Compare two next-token logit vectors."""
    baseline = baseline_logits.detach().float()
    variant = variant_logits.detach().float()
    if baseline.ndim == 1:
        baseline = baseline.unsqueeze(0)
    if variant.ndim == 1:
        variant = variant.unsqueeze(0)
    if baseline.shape != variant.shape:
        raise ValueError(f"logit shape mismatch: {tuple(baseline.shape)} vs {tuple(variant.shape)}")

    baseline_logp = torch.log_softmax(baseline, dim=-1)
    variant_logp = torch.log_softmax(variant, dim=-1)
    baseline_p = baseline_logp.exp()
    variant_p = variant_logp.exp()
    delta = variant - baseline
    top_baseline = int(torch.argmax(baseline[0]).item())
    top_variant = int(torch.argmax(variant[0]).item())
    return {
        "kl_baseline_to_variant": float((baseline_p * (baseline_logp - variant_logp)).sum(dim=-1).mean().item()),
        "kl_variant_to_baseline": float((variant_p * (variant_logp - baseline_logp)).sum(dim=-1).mean().item()),
        "max_abs_logit_delta": float(delta.abs().max().item()),
        "mean_abs_logit_delta": float(delta.abs().mean().item()),
        "top_token_baseline": top_baseline,
        "top_token_variant": top_variant,
        "top_token_changed": top_baseline != top_variant,
    }


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("at least one float is required")
    return parsed


def parse_controls(value: str) -> list[str]:
    controls = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(controls) - set(DEFAULT_CONTROLS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown control(s): {', '.join(unknown)}")
    if not controls:
        raise argparse.ArgumentTypeError("at least one control is required")
    return controls


def summarize_alpha_response(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if "kl_baseline_to_variant" not in record:
            continue
        by_control.setdefault(str(record.get("control")), []).append(record)

    def control_stats(control: str) -> dict[str, Any]:
        items = sorted(by_control.get(control, []), key=lambda item: float(item["alpha"]))
        values = [float(item["kl_baseline_to_variant"]) for item in items]
        alphas = [float(item["alpha"]) for item in items]
        return {
            "count": len(values),
            "alphas": alphas,
            "kl_min": min(values) if values else None,
            "kl_max": max(values) if values else None,
            "kl_range": (max(values) - min(values)) if values else None,
        }

    stats = {control: control_stats(control) for control in DEFAULT_CONTROLS}
    max_alpha = max((float(record["alpha"]) for record in records if "alpha" in record), default=None)

    def mean_kl(control: str, alpha: float | None) -> float | None:
        if alpha is None:
            return None
        values = [
            float(record["kl_baseline_to_variant"])
            for record in by_control.get(control, [])
            if float(record["alpha"]) == alpha
        ]
        return sum(values) / len(values) if values else None

    correct_at_max = mean_kl("correct", max_alpha)
    shuffled_at_max = mean_kl("shuffled", max_alpha)
    random_at_max = mean_kl("random_matched_norm", max_alpha)
    return {
        "by_control": stats,
        "max_alpha": max_alpha,
        "correct_non_flat": bool(stats["correct"]["kl_range"] is not None and stats["correct"]["kl_range"] > 1e-7),
        "correct_vs_shuffled_kl_gap_at_max_alpha": (
            correct_at_max - shuffled_at_max if correct_at_max is not None and shuffled_at_max is not None else None
        ),
        "correct_vs_random_kl_gap_at_max_alpha": (
            correct_at_max - random_at_max if correct_at_max is not None and random_at_max is not None else None
        ),
    }


def _ids_to_tensor(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _tokenizer_offsets(tokenizer: Any, rendered_text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    return [int(item) for item in input_ids], [(int(left), int(right)) for left, right in offsets]


def build_injection_prompt(
    tokenizer: Any,
    *,
    injection_text: str,
    marker_token_strategy: str,
) -> dict[str, Any]:
    content = AV_MARKER_PROMPT.replace("<NLA_ACTIVATION_MARKER>", injection_text)
    rendered = render_chat_prompt(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    offset_ids, offsets = _tokenizer_offsets(tokenizer, rendered.text or "")
    if offset_ids != rendered.token_ids:
        raise RuntimeError(
            "tokenizer offset pass did not match apply_chat_template(tokenize=True); "
            "cannot safely locate injection marker"
        )
    positions = marker_token_positions(
        rendered_text=rendered.text or "",
        offsets=offsets,
        marker_text=injection_text,
        left_context="<concept>",
        right_context="</concept>",
    )
    position = find_marker_token_position(
        rendered_text=rendered.text or "",
        offsets=offsets,
        marker_text=injection_text,
        left_context="<concept>",
        right_context="</concept>",
        strategy=marker_token_strategy,
    )
    input_ids, attention_mask = _ids_to_tensor(rendered.token_ids)
    return {
        "content": content,
        "rendered": rendered,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_text": injection_text,
        "marker_token_positions": positions,
        "injection_position": position,
    }


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


def forward_next_logits(
    model: Any,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
    logits = _extract_logits(output)
    return logits[:, -1, :].detach()


def select_token_vector(tensor: torch.Tensor, tau: int) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected tensor shape [1,T,d], got {tuple(tensor.shape)}")
    idx = tau if tau >= 0 else tensor.shape[1] + tau
    if not 0 <= idx < tensor.shape[1]:
        raise ValueError(f"tau={tau} resolves to {idx}, out of range for sequence length {tensor.shape[1]}")
    return tensor[0, idx].detach().clone()


def random_matched_norm(vector: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    return normalize_activation(random, float(vector.float().norm().item())).to(vector.dtype)


def source_vectors_from_prompts(
    *,
    model: Any,
    prompts: list[Any],
    boundaries: list[int],
    tau: int,
) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    for boundary_b in boundaries:
        for prompt in prompts:
            tensor, hook_path = capture_boundary_tensor(
                model=model,
                input_ids=prompt.input_ids,
                attention_mask=prompt.attention_mask,
                boundary_b=boundary_b,
            )
            vector = select_token_vector(tensor, tau)
            vectors.append(
                {
                    "boundary_b": boundary_b,
                    "prompt_name": prompt.name,
                    "hook_path": hook_path,
                    "tau": tau,
                    "resolved_tau": int(tensor.shape[1] + tau if tau < 0 else tau),
                    "token_count": int(tensor.shape[1]),
                    "vector": vector,
                    "vector_l2": float(vector.float().norm().item()),
                }
            )
    return vectors


def control_vector_for(
    *,
    source_index: int,
    sources: list[dict[str, Any]],
    control: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = sources[source_index]
    if control == "correct":
        return source["vector"], {"source_index": source_index}
    if control == "shuffled":
        same_boundary = [
            idx for idx, item in enumerate(sources)
            if idx != source_index and item["boundary_b"] == source["boundary_b"]
        ]
        if not same_boundary:
            raise ValueError(f"cannot make shuffled control for source_index={source_index}; no same-boundary peer")
        shuffled_index = same_boundary[0]
        return sources[shuffled_index]["vector"], {"source_index": shuffled_index}
    if control == "random_matched_norm":
        return random_matched_norm(source["vector"], seed=seed + source_index), {"source_index": None}
    raise ValueError(f"unknown control {control!r}")


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
    parser.add_argument("--embedding-equivalence-max-kl", type=float, default=1e-5)
    parser.add_argument("--embedding-equivalence-max-abs", type=float, default=1e-3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_track_a_probe.v1",
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "track": "A",
        "definition": "replace one input embedding row with scaled residual vector",
        "boundary_order": args.boundaries,
        "source_tau": args.source_tau,
        "requested_prompt_names": args.prompt_names,
        "alphas": args.alphas,
        "controls": args.controls,
        "injection_prompt": {},
        "embedding_equivalence": {},
        "source_vectors": [],
        "records": [],
        "summary": {},
        "path_passed": False,
        "signal_non_flat": False,
        "passed": False,
        "blockers": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    payload = payload_base(args, run_dir)
    output_path = run_dir / "track_a_probe.json"

    if torch is None:
        payload["blockers"] = [{"kind": "environment", "label": "torch import", "error": "PyTorch is required for Track A probe."}]
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
            "injection_position": injection_prompt["injection_position"],
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
    embeddings_module = resolved["embeddings"].obj
    if embeddings_module is None:
        blockers.append(classify_blocker("model load", f"could not resolve embeddings: {json_safe(resolved)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    payload["model"].update(
        {
            "hidden_size": get_config_value(config, "hidden_size"),
            "block_count": get_config_value(config, "num_hidden_layers"),
            "block_pattern": block_pattern_from_config(config, resolved["layers"].obj),
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
        base_embeds = embeddings_module(input_ids)
        baseline_logits = forward_next_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        embed_baseline_logits = forward_next_logits(
            model,
            inputs_embeds=base_embeds,
            attention_mask=attention_mask,
        )
        equivalence = next_token_logit_metrics(baseline_logits, embed_baseline_logits)
        payload["embedding_equivalence"] = equivalence
        embedding_path_ok = (
            equivalence["kl_baseline_to_variant"] <= args.embedding_equivalence_max_kl
            and equivalence["max_abs_logit_delta"] <= args.embedding_equivalence_max_abs
        )
        if not embedding_path_ok:
            blockers.append(
                {
                    "kind": "Track A input_embeds mismatch",
                    "label": "input_ids vs inputs_embeds baseline",
                    "error": json.dumps(equivalence, sort_keys=True),
                }
            )

        hidden_size = int(get_config_value(config, "hidden_size"))
        base_scale = math.sqrt(hidden_size)
        for source_index, source in enumerate(source_vectors):
            for control in args.controls:
                control_vector, control_meta = control_vector_for(
                    source_index=source_index,
                    sources=source_vectors,
                    control=control,
                    seed=args.random_seed,
                )
                for alpha in args.alphas:
                    target_scale = float(alpha) * base_scale
                    scaled = normalize_activation(control_vector, target_scale=target_scale)
                    injected_embeds = replace_embedding_row(
                        base_embeds,
                        position=int(injection_prompt["injection_position"]),
                        vector=scaled,
                    )
                    variant_logits = forward_next_logits(
                        model,
                        inputs_embeds=injected_embeds,
                        attention_mask=attention_mask,
                    )
                    metrics = next_token_logit_metrics(baseline_logits, variant_logits)
                    record = {
                        **metrics,
                        "boundary_b": source["boundary_b"],
                        "source_prompt_name": source["prompt_name"],
                        "source_index": source_index,
                        "source_tau": source["tau"],
                        "resolved_tau": source["resolved_tau"],
                        "source_token_count": source["token_count"],
                        "control": control,
                        "control_source_index": control_meta["source_index"],
                        "alpha": float(alpha),
                        "base_scale": base_scale,
                        "target_scale": target_scale,
                        "injected_vector_l2": float(scaled.float().norm().item()),
                    }
                    payload["records"].append(record)
                    write_json(output_path, payload)

        summary = summarize_alpha_response(payload["records"])
        payload["summary"] = summary
        payload["signal_non_flat"] = bool(summary["correct_non_flat"])
        payload["path_passed"] = not blockers and embedding_path_ok
        payload["passed"] = payload["path_passed"] and payload["signal_non_flat"]
    except Exception as exc:
        blockers.append(classify_blocker("Track A probe", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))

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
