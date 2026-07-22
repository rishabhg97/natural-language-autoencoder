#!/usr/bin/env python3
"""Write reusable NLA sidecars for prefix-key activation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _training_templates(contract: dict[str, Any]) -> dict[str, str]:
    templates = dict(contract.get("prompt_templates") or {})
    if "actor" not in templates and "av" in templates:
        templates["actor"] = templates["av"]
    if "critic" not in templates and "ar" in templates:
        templates["critic"] = templates["ar"]
    missing = [key for key in ("actor", "critic") if not templates.get(key)]
    if missing:
        raise ValueError(f"contract prompt_templates missing required keys: {missing}")
    return templates


def _base_extraction(base_meta: dict[str, Any], contract: dict[str, Any], layer: int) -> dict[str, Any]:
    extraction = dict(base_meta.get("extraction") or {})
    extraction["d_model"] = int(extraction["d_model"])
    extraction["layer_index"] = int(layer)
    extraction.setdefault("norm", "none")
    for key in ("mse_scale", "injection_scale"):
        value = (contract.get("extraction") or {}).get(key)
        if value is not None:
            extraction[key] = value
    return extraction


def _contract_tokens(contract: dict[str, Any], *, critic_suffix_ids: list[int] | None) -> dict[str, Any]:
    tokens = contract.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError("contract missing tokens mapping")
    out = dict(tokens)
    out["critic_suffix_ids"] = critic_suffix_ids
    return out


def _parent_datasets(base_meta: dict[str, Any], base_path: Path, contract_path: Path) -> list[str]:
    parent = base_meta.get("dataset_id") or str(base_path)
    return _unique([*(base_meta.get("parent_datasets") or []), parent, str(contract_path)])


def build_ar_sidecar(
    *,
    base_meta: dict[str, Any],
    contract: dict[str, Any],
    rows: int,
    base_path: Path,
    contract_path: Path,
    layer: int,
    slug: str,
) -> dict[str, Any]:
    tokens = _contract_tokens(
        contract,
        critic_suffix_ids=(contract.get("tokens") or {}).get("critic_suffix_ids"),
    )
    if not tokens.get("critic_suffix_ids"):
        raise ValueError("contract tokens missing critic_suffix_ids")

    sidecar: dict[str, Any] = {
        "kind": "nla_dataset",
        "schema_version": 1,
        "keep_debug_metadata": True,
        "dataset_id": f"nano30b_r{layer}_ar_sft_{slug}",
        "stage": "ar_sft",
        "row_count": int(rows),
        "created_by": "scripts.nano_prefix_dataset_pipeline",
        "created_at": base_meta.get("created_at", ""),
        "extraction": _base_extraction(base_meta, contract, layer),
        "parent_datasets": _parent_datasets(base_meta, base_path, contract_path),
        "critic": {"extraction_layer_index": int(layer)},
        "tokens": tokens,
        "prompt_templates": _training_templates(contract),
    }
    for key in ("sequence", "normalization", "api_summaries"):
        if key in contract:
            sidecar[key] = contract[key]
    return sidecar


def build_av_sidecar(
    *,
    base_meta: dict[str, Any],
    contract: dict[str, Any],
    rows: int,
    base_path: Path,
    contract_path: Path,
    layer: int,
    slug: str,
) -> dict[str, Any]:
    sidecar: dict[str, Any] = {
        "kind": "nla_dataset",
        "schema_version": 1,
        "keep_debug_metadata": True,
        "dataset_id": f"nano30b_r{layer}_av_sft_{slug}",
        "stage": "av_sft",
        "row_count": int(rows),
        "created_by": "scripts.nano_prefix_dataset_pipeline",
        "created_at": base_meta.get("created_at", ""),
        "extraction": _base_extraction(base_meta, contract, layer),
        "parent_datasets": _parent_datasets(base_meta, base_path, contract_path),
        "tokens": _contract_tokens(contract, critic_suffix_ids=None),
        "prompt_templates": _training_templates(contract),
    }
    for key in ("sequence", "normalization", "api_summaries"):
        if key in contract:
            sidecar[key] = contract[key]
    return sidecar


def write_ar_sidecar(
    *,
    base_path: Path,
    ar_path: Path,
    contract_path: Path,
    layer: int,
    slug: str,
) -> dict[str, Any]:
    base_meta = _read_yaml(Path(str(base_path) + ".nla_meta.yaml"))
    contract = _read_yaml(contract_path)
    rows = pq.ParquetFile(ar_path).metadata.num_rows
    sidecar = build_ar_sidecar(
        base_meta=base_meta,
        contract=contract,
        rows=rows,
        base_path=base_path,
        contract_path=contract_path,
        layer=layer,
        slug=slug,
    )
    Path(str(ar_path) + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
    return sidecar


def write_av_sidecar(
    *,
    base_path: Path,
    av_path: Path,
    contract_path: Path,
    layer: int,
    slug: str,
) -> dict[str, Any]:
    base_meta = _read_yaml(Path(str(base_path) + ".nla_meta.yaml"))
    contract = _read_yaml(contract_path)
    rows = pq.ParquetFile(av_path).metadata.num_rows
    sidecar = build_av_sidecar(
        base_meta=base_meta,
        contract=contract,
        rows=rows,
        base_path=base_path,
        contract_path=contract_path,
        layer=layer,
        slug=slug,
    )
    Path(str(av_path) + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
    return sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ar = subparsers.add_parser("ar", help="write an AR-SFT NLA dataset sidecar")
    ar.add_argument("--base", type=Path, required=True)
    ar.add_argument("--ar", type=Path, required=True)
    ar.add_argument("--contract", type=Path, required=True)
    ar.add_argument("--layer", type=int, required=True)
    ar.add_argument("--slug", required=True)
    av = subparsers.add_parser("av", help="write an AV-SFT NLA dataset sidecar")
    av.add_argument("--base", type=Path, required=True)
    av.add_argument("--av", type=Path, required=True)
    av.add_argument("--contract", type=Path, required=True)
    av.add_argument("--layer", type=int, required=True)
    av.add_argument("--slug", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "ar":
        sidecar = write_ar_sidecar(
            base_path=args.base,
            ar_path=args.ar,
            contract_path=args.contract,
            layer=args.layer,
            slug=args.slug,
        )
        print({"ar_sft": str(args.ar), "rows": sidecar["row_count"], "kind": sidecar["kind"]})
        return 0
    if args.command == "av":
        sidecar = write_av_sidecar(
            base_path=args.base,
            av_path=args.av,
            contract_path=args.contract,
            layer=args.layer,
            slug=args.slug,
        )
        print({"av_sft": str(args.av), "rows": sidecar["row_count"], "kind": sidecar["kind"]})
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
