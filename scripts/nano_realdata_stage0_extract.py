#!/usr/bin/env python3
"""Nano real-corpus Stage 0 activation extraction.

This is the Nano-specific equivalent of the reference NLA
`nla.datagen.stage0_extract` step. It writes the same base parquet schema and
sidecar shape, but uses the confirmed Nano `.backbone.layers` residual boundary
helpers from this pilot instead of the reference HF extractor assumptions.

It does not train Nano, run PEFT/LoRA, serve, run RL, or call a teacher model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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

try:
    import torch
except ModuleNotFoundError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if NLA_ROOT.exists() and str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))

from nano_extraction_identity import parse_boundaries, prefix_forward_to_R_b  # noqa: E402
from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    add_bool_optional_arg,
    classify_blocker,
    json_safe,
    load_config_from_args,
    load_model_from_args,
    load_tokenizer_from_args,
    write_json,
)

_MIN_POSITION = 50


def _schema(d_model: int, keep_token_metadata: bool) -> Any:
    fields = [
        ("n_raw_tokens", pa.int64()),
        ("detokenized_text_truncated", pa.string()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
    ]
    if keep_token_metadata:
        fields += [
            ("token_position", pa.int64()),
            ("token_id", pa.int64()),
            ("token_text", pa.string()),
            ("token_ids_prefix", pa.list_(pa.int32())),
        ]
    return pa.schema(fields)


def _dataset_id(base_model: str, revision: str | None, layer: int, corpus: str, corpus_slice: dict[str, int]) -> str:
    model_tag = base_model.split("/")[-1]
    rev_tag = revision or "unrev"
    digest = hashlib.sha256(f"{base_model}|{rev_tag}|{layer}|{corpus}|{corpus_slice}".encode()).hexdigest()[:8]
    return f"base_{model_tag}_L{layer}_{digest}"


def _sample_positions(
    token_ids: list[int], n_positions: int, special_ids: set[int], doc_id: str, seed: int
) -> list[int]:
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    candidates = [idx for idx, token_id in enumerate(token_ids) if idx >= _MIN_POSITION and token_id not in special_ids]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(n_positions, len(candidates)))


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _load_corpus(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset, load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets is required for real-corpus extraction") from exc

    kwargs: dict[str, Any] = {}
    if args.corpus_config:
        kwargs["name"] = args.corpus_config
    if args.corpus_revision:
        kwargs["revision"] = args.corpus_revision

    if args.streaming:
        ds = load_dataset(args.corpus, split=args.corpus_split, streaming=True, **kwargs)
        return list(ds.skip(args.corpus_start).take(args.corpus_length))

    ds = load_dataset(args.corpus, split=args.corpus_split, **kwargs)
    if not isinstance(ds, Dataset):
        raise TypeError(f"expected concrete Dataset for non-streaming mode, got {type(ds).__name__}")
    selected = ds.select(range(args.corpus_start, args.corpus_start + args.corpus_length))
    return [dict(row) for row in selected]


def _encode_batch(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, Any]:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _write_sidecar(args: argparse.Namespace, d_model: int, row_count: int, boundary_b: int) -> None:
    from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar_local

    corpus_slice = {"start": args.corpus_start, "length": args.corpus_length}
    meta = NLADatasetMeta(
        dataset_id=_dataset_id(args.model_id, args.model_revision, boundary_b, args.corpus, corpus_slice),
        stage="base",
        row_count=row_count,
        extraction=NLAExtractionMeta(
            base_model=args.model_id,
            d_model=d_model,
            layer_index=boundary_b,
            norm="none",
            corpus=args.corpus,
            corpus_slice=corpus_slice,
            positions_per_doc=args.positions_per_doc,
        ),
        created_by="scripts.nano_realdata_stage0_extract",
    )
    write_sidecar_local(args.output, meta)


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
    parser.add_argument("--boundary", type=parse_boundaries, default=[34], help="Single residual boundary, e.g. R_34.")
    parser.add_argument("--corpus", default="HuggingFaceFW/fineweb")
    parser.add_argument("--corpus-config", default="sample-10BT")
    parser.add_argument("--corpus-revision", default=None)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--corpus-start", type=int, default=0)
    parser.add_argument("--corpus-length", type=int, default=64)
    add_bool_optional_arg(parser, "--streaming", default=True)
    parser.add_argument("--positions-per-doc", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-token-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store exact token-position and prefix IDs for replay/debugging. Disable only for large production extracts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_path = args.metadata_output or args.output.with_suffix(args.output.suffix + ".metadata.json")
    payload: dict[str, Any] = {
        "schema_version": "nano_realdata_stage0_extract.v1",
        "output": str(args.output),
        "metadata_output": str(metadata_path),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "corpus": {
            "name": args.corpus,
            "config": args.corpus_config,
            "revision": args.corpus_revision,
            "split": args.corpus_split,
            "text_column": args.text_column,
            "start": args.corpus_start,
            "length": args.corpus_length,
            "streaming": args.streaming,
        },
        "boundary_b": args.boundary,
        "positions_per_doc": args.positions_per_doc,
        "keep_token_metadata": args.keep_token_metadata,
        "row_count": 0,
        "skipped": {"missing_text": 0, "too_short": 0, "short_sampled": 0},
        "blockers": [],
    }

    if torch is None or pa is None or pq is None:
        missing = [name for name, module in {"torch": torch, "pyarrow": pa}.items() if module is None]
        payload["blockers"] = [{"kind": "environment", "label": "imports", "error": f"missing modules: {missing}"}]
        write_json(metadata_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2
    if len(args.boundary) != 1:
        payload["blockers"] = [{"kind": "configuration", "label": "boundary", "error": "Stage 0 writes one boundary per parquet."}]
        write_json(metadata_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2
    boundary_b = int(args.boundary[0])

    try:
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            payload["blockers"].append(classify_blocker("remote-code load", config_error))
        model = load_model_from_args(args, config)
        model.eval()
        d_model = int(getattr(config, "hidden_size"))
        schema = _schema(d_model, args.keep_token_metadata)
        examples = _load_corpus(args)
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    except Exception as exc:
        payload["blockers"].append(classify_blocker("setup", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))
        write_json(metadata_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    try:
        with pq.ParquetWriter(args.output, schema) as writer, torch.no_grad():
            for chunk_start in range(0, len(examples), args.chunk_size):
                chunk = examples[chunk_start : chunk_start + args.chunk_size]
                texts: list[str] = []
                doc_indices: list[int] = []
                for offset, row in enumerate(chunk):
                    text = row.get(args.text_column)
                    if not isinstance(text, str) or not text:
                        payload["skipped"]["missing_text"] += 1
                        continue
                    texts.append(text)
                    doc_indices.append(args.corpus_start + chunk_start + offset)
                rows: dict[str, list[Any]] = {name: [] for name in schema.names}
                for batch_start in range(0, len(texts), args.batch_size):
                    batch_texts = texts[batch_start : batch_start + args.batch_size]
                    batch_doc_indices = doc_indices[batch_start : batch_start + args.batch_size]
                    encoded = _encode_batch(tokenizer, batch_texts, args.max_length)
                    input_ids = encoded["input_ids"].to(_model_start_device(model))
                    attention_mask = encoded["attention_mask"].to(input_ids.device)
                    residual = prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=boundary_b)

                    for batch_idx, doc_idx in enumerate(batch_doc_indices):
                        valid_len = int(attention_mask[batch_idx].sum().item())
                        token_ids = input_ids[batch_idx, :valid_len].detach().cpu().tolist()
                        doc_id = f"{args.corpus}:{args.corpus_split}:{doc_idx}"
                        positions = _sample_positions(token_ids, args.positions_per_doc, special_ids, doc_id, args.seed)
                        if not positions:
                            payload["skipped"]["too_short"] += 1
                            continue
                        if len(positions) < args.positions_per_doc:
                            payload["skipped"]["short_sampled"] += 1
                        for pos in positions:
                            truncated_ids = token_ids[: pos + 1]
                            rows["n_raw_tokens"].append(pos + 1)
                            rows["detokenized_text_truncated"].append(
                                tokenizer.decode(truncated_ids, skip_special_tokens=True)
                            )
                            rows["activation_vector"].append(residual[batch_idx, pos].detach().float().cpu().tolist())
                            rows["activation_layer"].append(boundary_b)
                            rows["doc_id"].append(doc_id)
                            if args.keep_token_metadata:
                                token_id = int(token_ids[pos])
                                rows["token_position"].append(pos)
                                rows["token_id"].append(token_id)
                                rows["token_text"].append(tokenizer.decode([token_id], skip_special_tokens=False))
                                rows["token_ids_prefix"].append([int(token_id) for token_id in truncated_ids])

                if rows["doc_id"]:
                    writer.write_table(pa.Table.from_pydict(rows, schema=schema))
                    row_count += len(rows["doc_id"])
                    payload["row_count"] = row_count
                    write_json(metadata_path, payload)

        _write_sidecar(args, d_model=d_model, row_count=row_count, boundary_b=boundary_b)
    except Exception as exc:
        payload["blockers"].append(
            classify_blocker("boundary extraction", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}")
        )
        write_json(metadata_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 1

    payload["row_count"] = row_count
    payload["sidecar"] = str(args.output) + ".nla_meta.yaml"
    write_json(metadata_path, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    print(f"\nwrote {row_count} rows -> {args.output}")
    return 0 if row_count > 0 and not payload["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
