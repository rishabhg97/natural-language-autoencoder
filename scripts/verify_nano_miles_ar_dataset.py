#!/usr/bin/env python3
"""Verify that a Nano AR-SFT parquet matches the Miles/NLA critic contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_nano_miles_av_dataset import (  # noqa: E402
    DEFAULT_SPLITS,
    _activation_lengths,
    _assert_no_content_cross_split_overlap,
    _content_column_order,
    _doc_split_report,
    _finite_counts,
    _row_content_key,
    _split_label,
    content_family_manifest_split_report,
    materialized_split_content_report,
    parse_split,
    sidecar_path_for,
)


TEXT_OPEN = "<text>"
TEXT_CLOSE = "</text>"


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    if isinstance(encoded, Mapping):
        ids = encoded["input_ids"]
    else:
        ids = encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"expected one tokenized prompt, got batch of {len(ids)}")
        ids = ids[0]
    return [int(tok) for tok in ids]


def _extract_explanation(prompt: str) -> str:
    start = prompt.find(TEXT_OPEN)
    end = prompt.find(TEXT_CLOSE, start + len(TEXT_OPEN))
    if start < 0 or end < 0:
        return ""
    return prompt[start + len(TEXT_OPEN) : end].strip()


def _validate_sidecar(sidecar: dict[str, Any], rows: int, expected_d_model: int | None) -> tuple[int, list[int]]:
    if sidecar.get("kind") != "nla_dataset":
        raise ValueError(f"sidecar kind must be nla_dataset, got {sidecar.get('kind')!r}")
    if sidecar.get("stage") != "ar_sft":
        raise ValueError(f"sidecar stage must be ar_sft, got {sidecar.get('stage')!r}")
    sidecar_rows = sidecar.get("row_count")
    if sidecar_rows is not None and int(sidecar_rows) != rows:
        raise ValueError(f"sidecar row_count {sidecar_rows} != parquet rows {rows}")
    extraction = sidecar.get("extraction") or {}
    d_model_value = expected_d_model or extraction.get("d_model")
    if not d_model_value:
        raise ValueError("sidecar extraction.d_model is required")
    d_model = int(d_model_value)
    if "layer_index" not in extraction:
        raise ValueError("sidecar extraction.layer_index is required")
    tokens = sidecar.get("tokens") or {}
    suffix_ids = tokens.get("critic_suffix_ids")
    if not isinstance(suffix_ids, list) or not suffix_ids:
        raise ValueError("sidecar tokens.critic_suffix_ids is required")
    templates = sidecar.get("prompt_templates") or {}
    critic_template = templates.get("critic")
    if not isinstance(critic_template, str) or "{explanation}" not in critic_template:
        raise ValueError("sidecar prompt_templates.critic with {explanation} is required")
    return d_model, [int(tok) for tok in suffix_ids]


def verify_dataset(
    parquet_path: str | Path,
    *,
    tokenizer: Any | None = None,
    tokenizer_model: str | None = None,
    expected_rows: int | None = None,
    expected_d_model: int | None = None,
    row_limit: int | None = None,
    prompt_check_limit: int | None = None,
    split_specs: tuple[tuple[float, float, float], ...] = DEFAULT_SPLITS,
    split_seed: int = 42,
    content_family_manifest: str | Path | None = None,
    content_family_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    parquet_path = Path(parquet_path)
    sidecar_path = sidecar_path_for(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")

    pf = pq.ParquetFile(parquet_path)
    if expected_rows is not None and pf.metadata.num_rows != expected_rows:
        raise ValueError(f"row count {pf.metadata.num_rows} != expected {expected_rows}")
    sidecar = yaml.safe_load(sidecar_path.read_text())
    expected_d_model, suffix_ids = _validate_sidecar(sidecar, pf.metadata.num_rows, expected_d_model)
    if tokenizer is None and tokenizer_model is not None:
        tokenizer = _load_tokenizer(tokenizer_model)

    columns = set(pf.schema_arrow.names)
    required = {"prompt", "activation_vector"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"parquet missing required columns: {missing}")

    length_counts: Counter[int] = Counter()
    finite_count = 0
    nonfinite_count = 0
    empty_explanation_rows: list[int] = []
    suffix_bad_rows: list[dict[str, Any]] = []
    doc_to_rows: dict[str, list[int]] = defaultdict(list)
    doc_to_content_keys: dict[str, set[str]] = defaultdict(set)
    content_columns_seen: set[str] = set()
    inspected = 0
    suffix_checked = 0

    for batch in pf.iter_batches(batch_size=4096):
        if row_limit is not None and inspected >= row_limit:
            break
        if row_limit is not None:
            batch = batch.slice(0, min(len(batch), row_limit - inspected))

        names = batch.schema.names
        av = batch.column(names.index("activation_vector"))
        lengths = _activation_lengths(av)
        length_counts.update(int(length) for length in lengths)
        finite, nonfinite = _finite_counts(av)
        finite_count += finite
        nonfinite_count += nonfinite

        prompts = batch.column(names.index("prompt")).to_pylist()
        docs = batch.column(names.index("doc_id")).to_pylist() if "doc_id" in names else [None] * len(batch)
        api_explanations = (
            batch.column(names.index("api_explanation")).to_pylist()
            if "api_explanation" in names
            else [None] * len(batch)
        )
        content_columns = _content_column_order(names)
        content_columns_seen.update(content_columns)
        content_values = {name: batch.column(names.index(name)).to_pylist() for name in content_columns}

        for offset, (prompt, doc_id, api_explanation) in enumerate(zip(prompts, docs, api_explanations)):
            row_index = inspected + offset
            doc_key = str(doc_id or f"__row_{row_index}")
            doc_to_rows[doc_key].append(row_index)
            if not isinstance(prompt, str) or not prompt:
                empty_explanation_rows.append(row_index)
                continue
            explanation = str(api_explanation).strip() if api_explanation is not None else _extract_explanation(prompt)
            doc_to_content_keys[doc_key].add(_row_content_key(content_values, offset, fallback=explanation or prompt))
            if not explanation:
                empty_explanation_rows.append(row_index)
            if tokenizer is not None and (prompt_check_limit is None or suffix_checked < prompt_check_limit):
                try:
                    ids = _token_ids(tokenizer, prompt)
                    n_suffix = len(suffix_ids)
                    if len(ids) < n_suffix or ids[-n_suffix:] != suffix_ids:
                        suffix_bad_rows.append(
                            {
                                "row_index": row_index,
                                "tail": ids[-n_suffix:] if len(ids) >= n_suffix else ids,
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - row-level diagnostics.
                    suffix_bad_rows.append({"row_index": row_index, "error": str(exc)})
                suffix_checked += 1

        inspected += len(batch)

    if set(length_counts) != {expected_d_model}:
        raise ValueError(f"activation_vector lengths {dict(length_counts)} != expected d_model {expected_d_model}")
    if nonfinite_count:
        raise ValueError(f"activation_vector contains {nonfinite_count} non-finite values")
    if empty_explanation_rows:
        raise ValueError(f"{len(empty_explanation_rows)} prompts have empty explanation text")
    if suffix_bad_rows:
        first = suffix_bad_rows[0]
        detail = first.get("error") or f"tail {first.get('tail')}"
        raise ValueError(
            f"{len(suffix_bad_rows)} prompts failed critic suffix check; "
            f"first row {first['row_index']}: {detail}"
        )

    if content_family_manifest is not None and split_specs:
        raise ValueError(
            "content_family_manifest and synthetic split_specs are mutually exclusive"
        )
    splits = {
        _split_label(spec): _doc_split_report(doc_to_rows, spec, split_seed, doc_to_content_keys)
        for spec in split_specs
    }
    _assert_no_content_cross_split_overlap(splits)
    report = {
        "parquet": str(parquet_path),
        "sidecar": str(sidecar_path),
        "row_count": pf.metadata.num_rows,
        "inspected_count": inspected,
        "sidecar_row_count": sidecar.get("row_count"),
        "stage": sidecar.get("stage"),
        "activation": {
            "d_model": expected_d_model,
            "length_counts": dict(length_counts),
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
        },
        "prompts": {
            "empty_explanation_count": len(empty_explanation_rows),
            "empty_explanation_rows_sample": empty_explanation_rows[:10],
        },
        "critic_suffix": {
            "checked_count": suffix_checked,
            "bad_count": len(suffix_bad_rows),
            "bad_rows_sample": suffix_bad_rows[:10],
            "tokenizer_check_skipped": tokenizer is None,
        },
        "content_hash": {
            "mode": "first_300_token_prefix_preferred",
            "columns_seen": sorted(content_columns_seen),
        },
        "splits": splits,
    }
    if content_family_manifest is not None:
        report["content_family_manifest_split"] = content_family_manifest_split_report(
            doc_to_rows,
            doc_to_content_keys,
            content_family_manifest,
            expected_sha256=content_family_manifest_sha256,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-d-model", type=int)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--prompt-check-limit", type=int)
    parser.add_argument("--tokenizer-model", help="HF tokenizer/model id or local model path for suffix checks.")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--split", dest="splits", action="append", type=parse_split)
    parser.add_argument(
        "--skip-synthetic-split-checks",
        action="store_true",
        help=(
            "Skip verifier-generated doc splits. Use this when validating an explicit "
            "materialized split layout such as content_component."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--content-family-manifest", type=Path)
    parser.add_argument("--content-family-manifest-sha256")
    parser.add_argument("--materialized-train", type=Path)
    parser.add_argument("--materialized-validation", type=Path)
    parser.add_argument("--materialized-test", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    tokenizer_model = None if args.skip_tokenizer_check else args.tokenizer_model
    if tokenizer_model is None and not args.skip_tokenizer_check:
        raise SystemExit("--tokenizer-model is required unless --skip-tokenizer-check is set")

    if args.content_family_manifest is not None and args.splits:
        parser.error("--content-family-manifest and --split are mutually exclusive")
    split_specs = (
        ()
        if args.skip_synthetic_split_checks or args.content_family_manifest is not None
        else tuple(args.splits or DEFAULT_SPLITS)
    )
    report = verify_dataset(
        args.parquet,
        tokenizer_model=tokenizer_model,
        expected_rows=args.expected_rows,
        expected_d_model=args.expected_d_model,
        row_limit=args.row_limit,
        prompt_check_limit=args.prompt_check_limit,
        split_specs=split_specs,
        split_seed=args.split_seed,
        content_family_manifest=args.content_family_manifest,
        content_family_manifest_sha256=args.content_family_manifest_sha256,
    )
    if args.skip_synthetic_split_checks or args.content_family_manifest is not None:
        report["synthetic_splits_skipped"] = True
    materialized_paths = {
        key: path
        for key, path in {
            "train": args.materialized_train,
            "validation": args.materialized_validation,
            "test": args.materialized_test,
        }.items()
        if path is not None
    }
    if materialized_paths:
        if set(materialized_paths) != {"train", "validation", "test"}:
            raise SystemExit("--materialized-train, --materialized-validation, and --materialized-test are all required")
        report["materialized_splits"] = materialized_split_content_report(materialized_paths)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
