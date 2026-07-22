#!/usr/bin/env python3
"""Build Nano Stage 3 NLA training parquets.

This is the Nano wrapper for the reference NLA Stage 3 contract. Stage 0 is
Nano-specific because it extracts confirmed Nano residual boundaries, but Stage
3 should stay reference-compatible: build `av_sft`, `ar_sft`, or `rl` parquets
and write sidecars with injection-token metadata and prompt templates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if NLA_ROOT.exists() and str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))

from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    add_bool_optional_arg,
    classify_blocker,
    json_safe,
    load_tokenizer_from_args,
    write_json,
)
from nla.datagen import injection_tokens  # noqa: E402
from nla.datagen.sidecar import read_sidecar_local, write_sidecar_local  # noqa: E402
from nla.schema import wrap_explanation  # noqa: E402


INJECT_PLACEHOLDER = "<INJECT>"
DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""
DEFAULT_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
CHUNK_SIZE = 4096
PROVENANCE_COLS = ["n_raw_tokens", "activation_layer", "doc_id"]
OPTIONAL_DEBUG_COLS = [
    "detokenized_text_truncated",
    "token_position",
    "token_id",
    "token_text",
    "token_ids_prefix",
]


def _debug_field(name: str) -> tuple[str, Any]:
    if name == "detokenized_text_truncated":
        return (name, pa.string())
    if name in ("token_position", "token_id"):
        return (name, pa.int64())
    if name == "token_text":
        return (name, pa.string())
    if name == "token_ids_prefix":
        return (name, pa.list_(pa.int32()))
    raise ValueError(f"unknown debug column: {name}")


def _prompt_struct_type() -> Any:
    return pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))


def _schema_for(stage: str, debug_columns: list[str], d_model: int) -> Any:
    activation = pa.list_(pa.float32(), d_model)
    prompt_struct = _prompt_struct_type()
    if stage == "av_sft":
        fields = [
            ("prompt", prompt_struct),
            ("response", pa.string()),
            ("activation_vector", activation),
        ]
    elif stage == "ar_sft":
        fields = [
            ("prompt", pa.string()),
            ("activation_vector", activation),
        ]
    elif stage == "rl":
        fields = [
            ("prompt", prompt_struct),
            ("activation_vector", activation),
        ]
    else:
        raise ValueError(f"unknown stage: {stage!r}")
    fields += [
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
    ]
    for col_name in debug_columns:
        fields.append(_debug_field(col_name))
    return pa.schema(fields)


def _tokenize_one(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _build_av_sft_cols(batch: Any, actor_prompt_content: str) -> dict[str, Any]:
    explanations = batch.column("api_explanation").to_pylist()
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    return {
        "prompt": pa.array([prompt_msg] * len(batch), type=_prompt_struct_type()),
        "response": pa.array([wrap_explanation(explanation) for explanation in explanations], type=pa.string()),
    }


def _build_rl_cols(batch: Any, actor_prompt_content: str) -> dict[str, Any]:
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    return {
        "prompt": pa.array([prompt_msg] * len(batch), type=_prompt_struct_type()),
    }


def _build_ar_sft_cols(batch: Any, critic_template: str, suffix_ids: list[int], tokenizer: Any) -> dict[str, Any]:
    prompts: list[str] = []
    suffix_len = len(suffix_ids)
    for explanation in batch.column("api_explanation").to_pylist():
        prompt = critic_template.format(explanation=explanation)
        ids = _tokenize_one(tokenizer, prompt)
        if len(ids) < suffix_len or ids[-suffix_len:] != suffix_ids:
            raise ValueError(
                f"critic prompt does not end with expected suffix IDs {suffix_ids}; "
                f"tail={ids[-suffix_len:] if len(ids) >= suffix_len else ids}"
            )
        prompts.append(prompt)
    return {"prompt": pa.array(prompts, type=pa.string())}


def _require_columns(column_names: list[str], required: set[str], stage: str) -> None:
    missing = sorted(required - set(column_names))
    if missing:
        raise ValueError(f"stage={stage} input is missing required columns: {missing}; available={column_names}")


def build_stage3(
    *,
    input_path: Path,
    output_path: Path,
    stage: str,
    tokenizer: Any,
    actor_template: str = DEFAULT_ACTOR_TEMPLATE,
    critic_template: str = DEFAULT_CRITIC_TEMPLATE,
    keep_debug_metadata: bool = True,
) -> dict[str, Any]:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for Nano Stage 3 build")
    if "{injection_char}" not in actor_template:
        raise ValueError("actor_template must contain {injection_char}")
    if stage == "ar_sft" and "{explanation}" not in critic_template:
        raise ValueError("critic_template must contain {explanation}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    in_meta = read_sidecar_local(input_path)
    if in_meta.stage != "base":
        raise ValueError(f"expected input sidecar stage='base', got {in_meta.stage!r}")
    if in_meta.extraction.norm != "none":
        raise ValueError(f"expected raw vectors norm='none', got {in_meta.extraction.norm!r}")

    critic_template_for_meta = critic_template if stage == "ar_sft" else None
    token_meta = injection_tokens.build_token_meta(
        tokenizer,
        actor_template,
        critic_template=critic_template_for_meta,
    )
    suffix_ids = token_meta.critic_suffix_ids
    actor_prompt_content = actor_template.format(injection_char=INJECT_PLACEHOLDER)

    in_pf = pq.ParquetFile(input_path)
    input_columns = in_pf.schema_arrow.names
    _require_columns(
        input_columns,
        {"activation_vector", "n_raw_tokens", "activation_layer", "doc_id"},
        stage,
    )
    if stage in ("av_sft", "ar_sft"):
        _require_columns(input_columns, {"api_explanation"}, stage)
    if in_pf.metadata.num_rows <= 0:
        raise ValueError(f"input parquet is empty: {input_path}")

    debug_cols = [col for col in OPTIONAL_DEBUG_COLS if keep_debug_metadata and col in input_columns]
    out_schema = _schema_for(stage, debug_cols, in_meta.extraction.d_model)
    carry_cols = PROVENANCE_COLS + debug_cols
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with pq.ParquetWriter(output_path, out_schema) as writer:
        for batch in in_pf.iter_batches(batch_size=CHUNK_SIZE):
            if stage == "av_sft":
                built = _build_av_sft_cols(batch, actor_prompt_content)
            elif stage == "ar_sft":
                if suffix_ids is None:
                    raise ValueError("ar_sft requires critic_suffix_ids")
                built = _build_ar_sft_cols(batch, critic_template, suffix_ids, tokenizer)
            elif stage == "rl":
                built = _build_rl_cols(batch, actor_prompt_content)
            else:
                raise ValueError(f"unknown stage: {stage!r}")

            for col in ["activation_vector", *carry_cols]:
                built[col] = batch.column(col)
            writer.write_table(pa.table(built, schema=out_schema))
            row_count += len(batch)

    out_meta = replace(
        in_meta,
        dataset_id=f"{stage}_{in_meta.dataset_id.removeprefix('base_')}",
        stage=stage,
        row_count=row_count,
        keep_debug_metadata=keep_debug_metadata,
        tokens=token_meta,
        prompt_templates={"actor": actor_template, "critic": critic_template},
        parent_datasets=[in_meta.dataset_id],
        created_by="scripts.nano_realdata_stage3_build",
        created_at="",
        git_commit="",
    )
    write_sidecar_local(output_path, out_meta)

    return {
        "stage": stage,
        "input": str(input_path),
        "output": str(output_path),
        "sidecar": str(output_path) + ".nla_meta.yaml",
        "row_count": row_count,
        "injection_char": token_meta.injection_char,
        "injection_token_id": token_meta.injection_token_id,
        "critic_suffix_ids": suffix_ids,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stage", choices=["av_sft", "ar_sft", "rl"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--actor-template", default=DEFAULT_ACTOR_TEMPLATE)
    parser.add_argument("--critic-template", default=DEFAULT_CRITIC_TEMPLATE)
    parser.add_argument("--keep-debug-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    parser.set_defaults(load_mode="meta", torch_dtype="auto", device_map="auto", attn_implementation=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest_output or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest: dict[str, Any] = {
        "schema_version": "nano_realdata_stage3_build.v1",
        "input": str(args.input),
        "stage": args.stage,
        "output": str(args.output),
        "manifest_output": str(manifest_path),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "blockers": [],
    }

    try:
        tokenizer = load_tokenizer_from_args(args)
        result = build_stage3(
            input_path=args.input,
            output_path=args.output,
            stage=args.stage,
            tokenizer=tokenizer,
            actor_template=args.actor_template,
            critic_template=args.critic_template,
            keep_debug_metadata=args.keep_debug_metadata,
        )
        manifest.update(result)
    except Exception as exc:
        manifest["blockers"].append(
            classify_blocker("stage3 build", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}")
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        return 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
    print(f"\nwrote {manifest['row_count']} rows ({args.stage}) -> {args.output}")
    return 0 if manifest.get("row_count", 0) > 0 and not manifest["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
