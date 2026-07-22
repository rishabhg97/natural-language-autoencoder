#!/usr/bin/env python3
"""Build AV-SFT parquet from a teacher-matched layer probe table.

The layer probe tables already contain the Nano residual activation vector and
the reused teacher explanation. This script writes the AV-SFT schema used by
Nano actor SFT: a constant injection prompt, wrapped explanation response, and
the activation/provenance columns.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
PROMPT_TYPE = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))


def wrap_explanation(text: Any) -> str:
    body = str(text or "").strip()
    return f"<explanation>\n{body}\n</explanation>"


def build_av_table(
    *,
    input_path: Path,
    source_sidecar: Path,
    output_path: Path,
    layer: int,
    batch_size: int = 4096,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not source_sidecar.exists():
        raise FileNotFoundError(source_sidecar)

    parquet = pq.ParquetFile(input_path)
    schema = parquet.schema_arrow
    required = {
        "api_explanation",
        "activation_vector",
        "n_raw_tokens",
        "activation_layer",
        "doc_id",
        "detokenized_text_truncated",
    }
    missing = sorted(required - set(schema.names))
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    activation_type = schema.field("activation_vector").type
    out_schema = pa.schema(
        [
            ("prompt", PROMPT_TYPE),
            ("response", pa.string()),
            ("activation_vector", activation_type),
            ("n_raw_tokens", pa.int64()),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
            ("detokenized_text_truncated", pa.string()),
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = [{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char="<INJECT>")}]
    row_count = 0
    empty_explanations = 0
    layer_values: set[int] = set()

    with pq.ParquetWriter(output_path, out_schema) as writer:
        for batch in parquet.iter_batches(batch_size=batch_size):
            explanations = batch.column("api_explanation").to_pylist()
            empty_explanations += sum(1 for value in explanations if not str(value or "").strip())
            layer_values.update(int(value) for value in batch.column("activation_layer").to_pylist())
            n = batch.num_rows
            writer.write_table(
                pa.table(
                    {
                        "prompt": pa.array([prompt] * n, type=PROMPT_TYPE),
                        "response": pa.array([wrap_explanation(value) for value in explanations], type=pa.string()),
                        "activation_vector": batch.column("activation_vector"),
                        "n_raw_tokens": batch.column("n_raw_tokens"),
                        "activation_layer": batch.column("activation_layer"),
                        "doc_id": batch.column("doc_id"),
                        "detokenized_text_truncated": batch.column("detokenized_text_truncated"),
                    },
                    schema=out_schema,
                )
            )
            row_count += n

    meta = yaml.safe_load(source_sidecar.read_text())
    parent_id = str(meta.get("dataset_id") or f"r{layer}_layer_probe")
    meta["dataset_id"] = f"nano30b_r{layer}_av_sft_start10500_len2048_teacher_reuse"
    meta["stage"] = "av_sft"
    meta["row_count"] = row_count
    meta["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["created_by"] = "scripts.nano_av_from_layer_probe"
    meta["parent_datasets"] = list(dict.fromkeys([*(meta.get("parent_datasets") or []), parent_id]))
    meta.pop("critic", None)
    tokens = dict(meta.get("tokens") or {})
    tokens["critic_suffix_ids"] = None
    meta["tokens"] = tokens
    prompts = dict(meta.get("prompt_templates") or {})
    prompts["actor"] = ACTOR_TEMPLATE
    prompts.setdefault("critic", CRITIC_TEMPLATE)
    meta["prompt_templates"] = prompts

    sidecar_path = Path(str(output_path) + ".nla_meta.yaml")
    sidecar_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sidecar_path": str(sidecar_path),
        "row_count": row_count,
        "empty_explanations": empty_explanations,
        "activation_layers": sorted(layer_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    report = build_av_table(
        input_path=args.input,
        source_sidecar=args.source_sidecar,
        output_path=args.output,
        layer=args.layer,
        batch_size=args.batch_size,
    )
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
