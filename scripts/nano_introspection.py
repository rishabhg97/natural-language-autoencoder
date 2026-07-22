#!/usr/bin/env python3
"""Nano30B architecture and tokenizer introspection harness.

This script is intentionally limited to environment/model metadata. It does not
train, serve, generate datasets, or run PEFT/RL. The companion
`nano_extraction_identity.py` performs the residual-boundary equality checks.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import inspect
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple


DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_OUTPUT_ROOT = Path("runs/introspection")
ROUTER_KEYS = (
    "num_experts_per_tok",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "routed_scaling_factor",
    "n_group",
    "topk_group",
)


class ResolvedModule(NamedTuple):
    path: str | None
    obj: Any | None
    confirmed: bool
    type_name: str | None


class RenderedPrompt(NamedTuple):
    text: str | None
    token_ids: list[int]
    sha256: str | None
    add_generation_prompt: bool
    enable_thinking_requested: bool
    enable_thinking_applied: bool
    template_error: str | None


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(output_root: Path, timestamp: str | None = None) -> Path:
    run_dir = output_root / (timestamp or utc_timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if hasattr(value, "shape"):
        return {"shape": [int(x) for x in value.shape], "dtype": str(getattr(value, "dtype", None))}
    return repr(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return {}


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def object_at_path(root: Any, dotted_path: str) -> Any | None:
    current = root
    for part in dotted_path.strip(".").split("."):
        if not part:
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def resolve_first(root: Any, candidates: tuple[str, ...], confirmed_path: str | None = None) -> ResolvedModule:
    for path in candidates:
        obj = object_at_path(root, path)
        if obj is not None:
            public_path = f".{path.strip('.')}"
            return ResolvedModule(
                path=public_path,
                obj=obj,
                confirmed=(public_path == confirmed_path if confirmed_path else True),
                type_name=type(obj).__name__,
            )
    return ResolvedModule(path=None, obj=None, confirmed=False, type_name=None)


def resolve_nano_module_paths(model: Any) -> dict[str, ResolvedModule]:
    backbone = resolve_first(model, ("backbone", "model", "transformer"), ".backbone")
    root = backbone.obj if backbone.obj is not None else model
    prefix = backbone.path or ""

    def nested(name: str, candidates: tuple[str, ...], confirmed_suffix: str) -> ResolvedModule:
        for candidate in candidates:
            obj = object_at_path(root, candidate)
            if obj is not None:
                path = f"{prefix}.{candidate}" if prefix else f".{candidate}"
                return ResolvedModule(path=path, obj=obj, confirmed=(path == f".backbone.{confirmed_suffix}"), type_name=type(obj).__name__)
        return ResolvedModule(path=None, obj=None, confirmed=False, type_name=None)

    return {
        "backbone": backbone,
        "layers": nested("layers", ("layers", "h", "blocks", "decoder.layers"), "layers"),
        "norm_f": nested("norm_f", ("norm_f", "norm", "final_layernorm", "ln_f"), "norm_f"),
        "embeddings": nested(
            "embeddings",
            ("embeddings", "embed_tokens", "wte", "word_embeddings"),
            "embeddings",
        ),
    }


def module_paths_json(resolved: dict[str, ResolvedModule]) -> dict[str, Any]:
    return {
        name: {
            "path": item.path,
            "confirmed": item.confirmed,
            "type": item.type_name,
        }
        for name, item in resolved.items()
    }


def nano_wrapper_assumptions_confirmed(resolved: dict[str, ResolvedModule]) -> bool:
    return all(
        resolved[name].path == expected
        for name, expected in {
            "backbone": ".backbone",
            "layers": ".backbone.layers",
            "norm_f": ".backbone.norm_f",
            "embeddings": ".backbone.embeddings",
        }.items()
    )


def block_pattern_from_config(config: Any, layers: Any | None = None) -> str | None:
    pattern = get_config_value(config, "hybrid_override_pattern")
    if isinstance(pattern, str):
        return pattern
    if layers is None:
        return None

    chars = []
    for layer in layers:
        block_type = str(getattr(layer, "block_type", "")).lower()
        class_hint = f"{type(layer).__name__} {type(getattr(layer, 'mixer', None)).__name__}".lower()
        if block_type == "mamba" or "mamba" in class_hint:
            chars.append("M")
        elif block_type == "attention" or "attention" in class_hint:
            chars.append("*")
        elif block_type == "moe" or "moe" in class_hint or "expert" in class_hint:
            chars.append("E")
        elif block_type in {"mlp", "ffn"} or "mlp" in class_hint:
            chars.append("-")
        else:
            chars.append("?")
    return "".join(chars)


def block_label(char: str) -> str:
    return {
        "M": "mamba",
        "*": "attention",
        "E": "moe",
        "-": "mlp",
        "?": "unknown",
    }.get(char, "unknown")


def block_table(pattern: str | None, layers: Any | None, layers_path: str | None) -> list[dict[str, Any]]:
    count = len(pattern) if pattern is not None else len(layers or [])
    rows = []
    for idx in range(count):
        char = pattern[idx] if pattern is not None and idx < len(pattern) else "?"
        layer = layers[idx] if layers is not None and idx < len(layers) else None
        rows.append(
            {
                "zero_based_block_index": idx,
                "residual_boundary_after_block": f"R_{idx + 1}",
                "pattern_char": char,
                "block_type": block_label(char),
                "module_path": f"{layers_path}.{idx}" if layers_path else None,
                "module_type": type(layer).__name__ if layer is not None else None,
            }
        )
    return rows


def router_config_from_config(config: Any) -> dict[str, Any]:
    data = {}
    for key in ROUTER_KEYS:
        value = get_config_value(config, key)
        if value is not None:
            data[key] = value
    return data


def tokenizer_metadata(tokenizer: Any, revision: str | None) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    return {
        "class": type(tokenizer).__name__ if tokenizer is not None else None,
        "revision": revision,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "chat_template_sha256": sha256_text(template) if isinstance(template, str) else None,
        "chat_template_present": isinstance(template, str) and bool(template),
        "enable_thinking_default": False,
    }


def _ids_to_list(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"expected one prompt, got batch of {len(ids)}")
        ids = ids[0]
    return [int(x) for x in ids]


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> tuple[Any, bool, str | None]:
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs), True, None
    except TypeError as exc:
        kwargs.pop("enable_thinking")
        try:
            return tokenizer.apply_chat_template(messages, **kwargs), False, str(exc)
        except Exception:
            raise


def render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> RenderedPrompt:
    text = None
    token_ids: list[int] = []
    applied = False
    error = None

    text_result, applied_text, error_text = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    applied = applied_text
    error = error_text
    if isinstance(text_result, str):
        text = text_result
    else:
        text = str(text_result)

    ids_result, applied_ids, error_ids = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    applied = applied and applied_ids
    error = error or error_ids
    token_ids = _ids_to_list(ids_result)

    return RenderedPrompt(
        text=text,
        token_ids=token_ids,
        sha256=sha256_text(text),
        add_generation_prompt=add_generation_prompt,
        enable_thinking_requested=(enable_thinking is False),
        enable_thinking_applied=applied,
        template_error=error,
    )


def torch_dtype_from_name(name: str | None) -> Any:
    if name is None or name == "auto":
        return "auto"
    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"unsupported dtype {name!r}; use auto, bfloat16, float16, or float32")
    return aliases[name]


@contextmanager
def init_empty_weights_context():
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            yield
        return
    except ModuleNotFoundError:
        pass

    import torch

    with torch.device("meta"):
        yield


def load_tokenizer_from_args(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    revision = args.tokenizer_revision or args.model_revision
    return AutoTokenizer.from_pretrained(
        args.model_id,
        revision=revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


def _raw_config_fallback(model_id: str, revision: str | None, local_files_only: bool) -> tuple[Any | None, str | None]:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
            local_files_only=local_files_only,
        )
        return SimpleNamespace(**json.loads(Path(path).read_text())), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_config_from_args(args: argparse.Namespace) -> tuple[Any | None, str | None]:
    try:
        from transformers import AutoConfig

        return (
            AutoConfig.from_pretrained(
                args.model_id,
                revision=args.model_revision,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            ),
            None,
        )
    except Exception as exc:
        fallback, fallback_error = _raw_config_fallback(args.model_id, args.model_revision, args.local_files_only)
        error = f"{type(exc).__name__}: {exc}"
        if fallback is not None:
            return fallback, f"{error}; raw config fallback used"
        return None, f"{error}; raw config fallback failed: {fallback_error}"


def load_model_from_args(args: argparse.Namespace, config: Any | None = None) -> Any:
    from transformers import AutoModelForCausalLM

    dtype = torch_dtype_from_name(args.torch_dtype)
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        common_kwargs["attn_implementation"] = args.attn_implementation

    if args.load_mode == "meta":
        if config is None or isinstance(config, SimpleNamespace):
            raise RuntimeError("meta load requires a transformers config object; raw config fallback is insufficient")
        with init_empty_weights_context():
            return AutoModelForCausalLM.from_config(config, **common_kwargs)

    kwargs = {
        **common_kwargs,
        "local_files_only": args.local_files_only,
        "revision": args.model_revision,
        "torch_dtype": dtype,
    }
    if args.device_map and args.device_map != "none":
        kwargs["device_map"] = args.device_map
        # Device-mapped inference models should stream checkpoint tensors into
        # their assigned devices instead of materializing a full CPU copy first.
        if getattr(args, "low_cpu_mem_usage", True):
            kwargs["low_cpu_mem_usage"] = True
    if config is not None and not isinstance(config, SimpleNamespace):
        kwargs["config"] = config
    return AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs).eval()


def classify_blocker(label: str, error: str) -> dict[str, str]:
    lower = error.lower()
    if "out of memory" in lower or "not enough memory" in lower or "mps" in lower:
        kind = "memory"
    elif "remote" in lower or "trust_remote_code" in lower or "mamba" in lower or "causal_conv" in lower:
        kind = "remote-code load"
    elif "chat_template" in lower or "enable_thinking" in lower:
        kind = "template ambiguity"
    elif "cache" in lower:
        kind = "cache API ambiguity"
    else:
        kind = label
    return {"kind": kind, "label": label, "error": error}


def cache_metadata_from_model(model: Any, config: Any) -> dict[str, Any]:
    if model is None:
        return {"accessible": False, "reason": "model not loaded"}
    try:
        module = sys.modules.get(type(model).__module__)
        cache_cls = getattr(module, "HybridMambaAttentionDynamicCache", None) if module is not None else None
        if cache_cls is None:
            return {"accessible": False, "reason": "HybridMambaAttentionDynamicCache not found in model module"}

        fields: dict[str, Any] = {}
        cache = None
        try:
            import torch

            cache = cache_cls(config, batch_size=1, dtype=torch.bfloat16, device="cpu")
        except Exception as exc:
            return {
                "accessible": True,
                "class": f"{cache_cls.__module__}.{cache_cls.__name__}",
                "signature": str(inspect.signature(cache_cls)),
                "instantiated": False,
                "instantiation_error": f"{type(exc).__name__}: {exc}",
            }

        for name in ("key_cache", "value_cache", "conv_states", "ssm_states", "transformer_layers", "hybrid_override_pattern"):
            if hasattr(cache, name):
                value = getattr(cache, name)
                if isinstance(value, list):
                    fields[name] = {
                        "type": "list",
                        "length": len(value),
                        "first": json_safe(value[0]) if value else None,
                    }
                else:
                    fields[name] = json_safe(value)
        return {
            "accessible": True,
            "class": f"{cache_cls.__module__}.{cache_cls.__name__}",
            "signature": str(inspect.signature(cache_cls)),
            "instantiated": True,
            "fields": fields,
        }
    except Exception as exc:
        return {
            "accessible": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def build_metadata_record(
    args: argparse.Namespace,
    *,
    tokenizer: Any | None,
    config: Any | None,
    model: Any | None,
    blockers: list[dict[str, str]],
    run_dir: Path,
) -> dict[str, Any]:
    resolved = resolve_nano_module_paths(model) if model is not None else {}
    layers = resolved.get("layers").obj if resolved else None
    pattern = block_pattern_from_config(config, layers)
    hidden_size = get_config_value(config, "hidden_size")
    block_count = get_config_value(config, "num_hidden_layers")
    if block_count is None and pattern is not None:
        block_count = len(pattern)

    module_paths = module_paths_json(resolved) if resolved else {
        "backbone": {"path": None, "confirmed": False, "type": None},
        "layers": {"path": None, "confirmed": False, "type": None},
        "norm_f": {"path": None, "confirmed": False, "type": None},
        "embeddings": {"path": None, "confirmed": False, "type": None},
    }

    return {
        "schema_version": "nano_introspection.v1",
        "created_at_utc": utc_timestamp(),
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "load_mode": args.load_mode,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "class": type(model).__name__ if model is not None else None,
        },
        "tokenizer": tokenizer_metadata(tokenizer, args.tokenizer_revision or args.model_revision) if tokenizer else None,
        "architecture": {
            "hidden_size": hidden_size,
            "block_count": block_count,
            "block_pattern": pattern,
            "module_paths": module_paths,
            "nano_wrapper_assumptions_confirmed": nano_wrapper_assumptions_confirmed(resolved) if resolved else False,
            "blocks": block_table(pattern, layers, resolved.get("layers").path if resolved else None),
        },
        "router": router_config_from_config(config),
        "cache": cache_metadata_from_model(model, config) if model is not None else {"accessible": False, "reason": "model not loaded"},
        "blockers": blockers,
    }


def collect_metadata(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    tokenizer = None
    config = None
    model = None

    try:
        tokenizer = load_tokenizer_from_args(args)
    except Exception as exc:
        blockers.append(classify_blocker("tokenizer load", f"{type(exc).__name__}: {exc}"))

    config, config_error = load_config_from_args(args)
    if config_error is not None:
        blockers.append(classify_blocker("remote-code load", config_error))

    if args.load_mode != "config":
        try:
            model = load_model_from_args(args, config)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
            blockers.append(classify_blocker("model load", error))

    return build_metadata_record(
        args,
        tokenizer=tokenizer,
        config=config,
        model=model,
        blockers=blockers,
        run_dir=run_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--load-mode", choices=("full", "meta", "config"), default="full")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args(argv)


def add_bool_optional_arg(parser: argparse.ArgumentParser, name: str, *, default: bool) -> None:
    """Python 3.8-compatible equivalent of argparse.BooleanOptionalAction."""
    dest = name.lstrip("-").replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        return
    parser.add_argument(name, dest=dest, action="store_true", default=default)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    metadata = collect_metadata(args, run_dir)
    out_path = run_dir / "metadata.json"
    write_json(out_path, metadata)
    print(json.dumps(json_safe(metadata), indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    return 0 if not metadata["blockers"] or args.load_mode == "config" else 2


if __name__ == "__main__":
    raise SystemExit(main())
