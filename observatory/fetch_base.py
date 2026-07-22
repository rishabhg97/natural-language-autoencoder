#!/usr/bin/env python3
"""Fetch and verify the public Nano30B base checkpoint for functional evals."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    resolve_path,
    write_json,
)
from .model_runtime import hf_checkpoint_complete


SCHEMA_VERSION = "nano_viz_base_fetch.v1"


def configure_hf_transport() -> dict[str, str]:
    """Use the proxy-compatible HTTP path unless the operator overrides it."""

    defaults = {
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "HF_HUB_ETAG_TIMEOUT": "60",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in defaults}


def download_snapshot_with_retries(
    download: object,
    *,
    repo_id: str,
    revision: str,
    cache_dir: str,
    max_workers: int,
    attempts: int = 5,
    sleep: object = time.sleep,
) -> Path:
    """Resume transiently failed public downloads with decreasing concurrency."""

    if attempts < 1 or max_workers < 1:
        raise ValueError("download attempts and max_workers must be positive")
    last_error: Exception | None = None
    for attempt in range(attempts):
        workers = max(1, max_workers // (2**attempt))
        try:
            resolved = download(  # type: ignore[operator]
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                max_workers=workers,
            )
            return Path(str(resolved))
        except Exception as exc:  # Network transports expose different exception classes.
            last_error = exc
            if attempt + 1 == attempts:
                break
            sleep(min(60.0, 5.0 * (2**attempt)))  # type: ignore[operator]
    assert last_error is not None
    raise OSError(f"snapshot download failed after {attempts} attempts: {last_error}") from last_error


def run(config_path: Path) -> dict:
    transport = configure_hf_transport()
    from huggingface_hub import snapshot_download
    from .prepare_runtime import stage_hf_checkpoint

    config = load_config(config_path)
    paths = config["paths"]
    models = config["models"]
    source = resolve_path(models["base_hf_source"], config_path=config_path)
    staged = resolve_path(models["base_hf"], config_path=config_path)
    if not hf_checkpoint_complete(source):
        resolved = download_snapshot_with_retries(
            snapshot_download,
            repo_id=str(models["base_repo_id"]),
            revision=str(models["base_revision"]),
            cache_dir=str(models["base_cache_dir"]),
            max_workers=int(models.get("base_download_workers", 8)),
        )
    else:
        resolved = source
    if resolved.resolve() != source.resolve():
        raise ObservatoryConfigError(
            f"downloaded base snapshot differs from configured path: {resolved} != {source}"
        )
    if not hf_checkpoint_complete(source):
        raise ObservatoryConfigError(f"base checkpoint remains incomplete: {source}")
    stage = stage_hf_checkpoint(
        source_checkpoint=source,
        output_hf=staged,
        force=False,
        copy_workers=int(config["evaluation"].get("stage_copy_workers", 8)),
    )
    weight_files = sorted(source.glob("*.safetensors"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "config_sha256": config_fingerprint(config),
        "repo_id": str(models["base_repo_id"]),
        "revision": str(models["base_revision"]),
        "snapshot": str(source),
        "staged_snapshot": str(staged),
        "stage": stage,
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "transport": {
            "xet_disabled": transport["HF_HUB_DISABLE_XET"] == "1",
            "download_timeout_seconds": int(transport["HF_HUB_DOWNLOAD_TIMEOUT"]),
            "etag_timeout_seconds": int(transport["HF_HUB_ETAG_TIMEOUT"]),
        },
    }
    output = (
        resolve_path(paths["model_outputs_dir"], config_path=config_path)
        / "runtime"
        / "base_fetch_report.json"
    )
    write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.config)
    except (OSError, ValueError, ObservatoryConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
