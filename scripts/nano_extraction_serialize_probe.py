#!/usr/bin/env python3
"""Tiny Nano30B residual-boundary extraction serialization probe.

This script loads the frozen Nano model, extracts a few R_b residual tensors
with forward hooks, writes them to disk, reloads them, and records a JSON
manifest. It does not train, generate datasets, serve, run PEFT, or run RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    build_prompt_inputs,
    parse_boundaries,
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
    resolve_nano_module_paths,
    write_json,
)


DEFAULT_PROMPT_NAMES = ("raw", "reasoning_off_chat", "av_marker", "ar_critic")


def parse_prompt_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one prompt name is required")
    unknown = sorted(set(names) - set(DEFAULT_PROMPT_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown prompt name(s): {', '.join(unknown)}")
    return names


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _move_inputs_to_model_start(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = _model_start_device(model)
    if device.type == "meta":
        raise RuntimeError("extraction serialization requires real weights; model is on meta device")
    return input_ids.to(device), attention_mask.to(device)


def _capture_hook_output(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    return tensor.detach().clone()


def capture_boundary_tensor(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_b: int,
) -> tuple[torch.Tensor, str]:
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    embeddings = resolved["embeddings"].obj
    if layers is None or embeddings is None:
        raise RuntimeError(f"could not resolve layers/embeddings: {json_safe(resolved)}")
    if not 0 <= boundary_b <= len(layers):
        raise ValueError(f"boundary_b={boundary_b} out of range for {len(layers)} blocks")

    input_ids, attention_mask = _move_inputs_to_model_start(model, input_ids, attention_mask)
    captured: dict[str, torch.Tensor] = {}

    if boundary_b == 0:
        return embeddings(input_ids).detach().clone(), resolved["embeddings"].path or ".backbone.embeddings"

    hook_path = f"{resolved['layers'].path}.{boundary_b - 1}"
    handle = layers[boundary_b - 1].register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("tensor", _capture_hook_output(output))
    )
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
    finally:
        handle.remove()

    if "tensor" not in captured:
        raise RuntimeError(f"forward hook did not fire for {hook_path}")
    return captured["tensor"], hook_path


def serialize_tensor_record(
    *,
    tensor: torch.Tensor,
    run_dir: Path,
    boundary_b: int,
    prompt_name: str,
    hook_path: str,
    token_count: int,
    save_tensors: bool,
) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu().contiguous()
    record: dict[str, Any] = {
        "prompt_name": prompt_name,
        "boundary_b": boundary_b,
        "hook_path": hook_path,
        "token_count": token_count,
        "shape": [int(dim) for dim in cpu_tensor.shape],
        "dtype": str(cpu_tensor.dtype),
        "saved": bool(save_tensors),
        "tensor_path": None,
        "tensor_sha256": None,
        "tensor_file_bytes": None,
        "reload_equal": None,
    }
    if not save_tensors:
        return record

    tensor_path = run_dir / f"R{boundary_b}_{safe_stem(prompt_name)}.pt"
    torch.save(
        {
            "tensor": cpu_tensor,
            "metadata": {
                "prompt_name": prompt_name,
                "boundary_b": boundary_b,
                "hook_path": hook_path,
                "token_count": token_count,
            },
        },
        tensor_path,
    )
    reloaded = torch.load(tensor_path, map_location="cpu", weights_only=True)
    reloaded_tensor = reloaded["tensor"] if isinstance(reloaded, dict) and "tensor" in reloaded else reloaded

    record.update(
        {
            "tensor_path": str(tensor_path),
            "tensor_sha256": file_sha256(tensor_path),
            "tensor_file_bytes": tensor_path.stat().st_size,
            "reload_equal": bool(torch.equal(cpu_tensor, reloaded_tensor)),
        }
    )
    return record


def probe_payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_extraction_serialize_probe.v1",
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "boundary_order": args.boundaries,
        "requested_prompt_names": args.prompt_names,
        "prompt_modes": [],
        "records": [],
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
    parser.add_argument("--prompt-names", type=parse_prompt_names, default=list(DEFAULT_PROMPT_NAMES))
    parser.add_argument("--prompt-max-length", type=int, default=256)
    add_bool_optional_arg(parser, "--save-tensors", default=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    manifest = probe_payload_base(args, run_dir)
    manifest_path = run_dir / "extraction_probe.json"

    if torch is None:
        manifest["blockers"] = [
            {
                "kind": "environment",
                "label": "torch import",
                "error": "PyTorch is required for extraction serialization but is not installed.",
            }
        ]
        write_json(manifest_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        print(f"\nwrote {manifest_path}")
        return 2

    blockers: list[dict[str, str]] = []
    try:
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
        prompts = build_prompt_inputs(tokenizer, args.prompt_max_length)
        prompts = [item for item in prompts if item.name in args.prompt_names]
        manifest["prompt_modes"] = [{"name": item.name, **item.metadata} for item in prompts]
    except Exception as exc:
        blockers.append(classify_blocker("tokenizer/config/template", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        manifest["blockers"] = blockers
        write_json(manifest_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        print(f"\nwrote {manifest_path}")
        return 2

    try:
        model = load_model_from_args(args, config)
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        manifest["blockers"] = blockers
        write_json(manifest_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        print(f"\nwrote {manifest_path}")
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
    manifest["model"].update(
        {
            "hidden_size": get_config_value(config, "hidden_size"),
            "block_count": get_config_value(config, "num_hidden_layers"),
            "block_pattern": block_pattern_from_config(config, layers),
        }
    )

    model.eval()
    for boundary_b in args.boundaries:
        for prompt in prompts:
            try:
                tensor, hook_path = capture_boundary_tensor(
                    model=model,
                    input_ids=prompt.input_ids,
                    attention_mask=prompt.attention_mask,
                    boundary_b=boundary_b,
                )
                record = serialize_tensor_record(
                    tensor=tensor,
                    run_dir=run_dir,
                    boundary_b=boundary_b,
                    prompt_name=prompt.name,
                    hook_path=hook_path,
                    token_count=int(prompt.input_ids.shape[-1]),
                    save_tensors=args.save_tensors,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
                blockers.append(classify_blocker("boundary serialization", error))
                record = {
                    "prompt_name": prompt.name,
                    "boundary_b": boundary_b,
                    "passed": False,
                    "error": error,
                }
            manifest["records"].append(record)
            write_json(manifest_path, manifest)

    manifest["blockers"] = blockers
    manifest["passed"] = bool(manifest["records"]) and not blockers and all(
        item.get("reload_equal", True) is not False for item in manifest["records"]
    )
    write_json(manifest_path, manifest)
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
    print(f"\nwrote {manifest_path}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
