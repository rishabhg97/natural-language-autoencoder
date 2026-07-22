#!/usr/bin/env python3
"""Build a Nano AR-SFT smoke parquet from Stage 2 explained rows.

Input is a reference-compatible `*_explained.parquet` with `api_explanation`
and `activation_vector`. Output is the minimal AR training schema consumed by
`scripts/nano_ar_frozen_baseline.py --ar-sft-parquet`.

This avoids AV injection-token metadata while the pilot is only testing the AR
contract: explanation text z -> frozen Nano prefix at R_b -> value head -> h_b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_ar_frozen_baseline import DEFAULT_CRITIC_TEMPLATE  # noqa: E402
from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    add_bool_optional_arg,
    classify_blocker,
    json_safe,
    load_tokenizer_from_args,
    write_json,
)


def _tokenize_one(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def compute_critic_suffix_ids(tokenizer: Any, critic_template: str) -> list[int]:
    if "{explanation}" not in critic_template:
        raise ValueError("critic-template must contain {explanation}")
    suffix_str = critic_template.split("{explanation}")[-1]
    suffix_ids = _tokenize_one(tokenizer, suffix_str)
    if len(suffix_ids) < 2:
        raise ValueError(
            f"critic suffix {suffix_str!r} tokenized to {len(suffix_ids)} tokens; need at least 2 "
            "so the BPE-boundary token can be dropped"
        )
    return suffix_ids[1:]


def verify_prompt_suffix(tokenizer: Any, prompt: str, suffix_ids: list[int]) -> None:
    ids = _tokenize_one(tokenizer, prompt)
    n_suffix = len(suffix_ids)
    if len(ids) < n_suffix or ids[-n_suffix:] != suffix_ids:
        raise ValueError(
            f"critic prompt does not end with expected suffix IDs {suffix_ids}; "
            f"tail={ids[-n_suffix:] if len(ids) >= n_suffix else ids}"
        )


def _copy_column(table: Any, name: str) -> Any | None:
    return table.column(name) if name in table.column_names else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Stage 2 *_explained.parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output AR-SFT parquet")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--critic-template", default=DEFAULT_CRITIC_TEMPLATE)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--verify-suffix", action=argparse.BooleanOptionalAction, default=True)
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
        "schema_version": "nano_realdata_ar_build.v1",
        "input": str(args.input),
        "output": str(args.output),
        "manifest_output": str(manifest_path),
        "critic_template": args.critic_template,
        "max_records": args.max_records,
        "verify_suffix": args.verify_suffix,
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "row_count": 0,
        "critic_suffix_ids": None,
        "blockers": [],
    }

    if pa is None or pq is None:
        manifest["blockers"] = [{"kind": "environment", "label": "pyarrow import", "error": "pyarrow is required"}]
        write_json(manifest_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        return 2

    try:
        tokenizer = load_tokenizer_from_args(args)
        suffix_ids = compute_critic_suffix_ids(tokenizer, args.critic_template)
        manifest["critic_suffix_ids"] = suffix_ids

        table = pq.read_table(args.input)
        names = set(table.column_names)
        required = {"api_explanation", "activation_vector"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{args.input} is missing required columns: {missing}")

        if args.max_records is not None:
            table = table.slice(0, args.max_records)
        explanations = table.column("api_explanation").to_pylist()
        prompts = [args.critic_template.format(explanation=explanation) for explanation in explanations]
        if args.verify_suffix:
            for prompt in prompts:
                verify_prompt_suffix(tokenizer, prompt, suffix_ids)

        output_cols: dict[str, Any] = {
            "prompt": pa.array(prompts, type=pa.string()),
            "activation_vector": table.column("activation_vector"),
        }
        for col_name in [
            "n_raw_tokens",
            "activation_layer",
            "doc_id",
            "detokenized_text_truncated",
            "token_position",
            "token_id",
            "token_text",
            "token_ids_prefix",
            "api_explanation",
        ]:
            column = _copy_column(table, col_name)
            if column is not None:
                output_cols[col_name] = column

        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output = pa.table(output_cols)
        pq.write_table(output, args.output)
        manifest["row_count"] = int(output.num_rows)
        manifest["columns"] = output.column_names
    except Exception as exc:
        manifest["blockers"].append(classify_blocker("ar build", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))
        write_json(manifest_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        return 1

    write_json(manifest_path, manifest)
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
    print(f"\nwrote {manifest['row_count']} rows -> {args.output}")
    return 0 if manifest["row_count"] > 0 and not manifest["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
