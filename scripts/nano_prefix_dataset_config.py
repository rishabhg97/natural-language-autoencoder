#!/usr/bin/env python3
"""Render or run the config-driven exact-prefix dataset pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_prefix_dataset_pipeline.v1"


class PrefixDatasetConfigError(ValueError):
    """Raised when a prefix dataset config is incomplete or unsafe."""


def _required(mapping: dict[str, Any], key: str, *, section: str) -> Any:
    value = mapping.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PrefixDatasetConfigError(f"{section}.{key} is required")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise PrefixDatasetConfigError("config must be a YAML mapping")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise PrefixDatasetConfigError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )
    paths = config.get("paths") or {}
    dataset = config.get("dataset") or {}
    extraction = config.get("extraction") or {}
    critic_initialization = config.get("critic_initialization") or {}
    build = config.get("build") or {}
    if not isinstance(paths, dict) or not isinstance(dataset, dict) or not isinstance(
        extraction,
        dict,
    ) or not isinstance(critic_initialization, dict):
        raise PrefixDatasetConfigError(
            "paths, dataset, extraction, and critic_initialization must be mappings"
        )
    for key in ("code_root", "model", "source_parquet", "contract", "out"):
        _required(paths, key, section="paths")
    for key in ("layer", "slug", "expected_rows", "expected_d_model"):
        _required(dataset, key, section="dataset")
    if int(dataset["expected_rows"]) <= 0 or int(dataset["expected_d_model"]) <= 0:
        raise PrefixDatasetConfigError(
            "dataset.expected_rows and expected_d_model must be positive"
        )
    expected_source_sha256 = str(
        dataset.get("expected_source_parquet_sha256") or ""
    )
    if expected_source_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", expected_source_sha256
    ):
        raise PrefixDatasetConfigError(
            "dataset.expected_source_parquet_sha256 must be 64 lowercase hex characters"
        )
    if bool(extraction.get("publication_mode", False)):
        _required(paths, "model_fingerprint_json", section="paths")
        _required(paths, "runtime_provenance_json", section="paths")
        _required(
            dataset,
            "expected_source_parquet_sha256",
            section="dataset",
        )
        if extraction.get("deterministic_algorithms") is not True:
            raise PrefixDatasetConfigError(
                "extraction.deterministic_algorithms must be true for publication"
            )
        if extraction.get("allow_tf32") is not False:
            raise PrefixDatasetConfigError(
                "extraction.allow_tf32 must be false for publication"
            )
        if extraction.get("cudnn_benchmark") is not False:
            raise PrefixDatasetConfigError(
                "extraction.cudnn_benchmark must be false for publication"
            )
        if str(extraction.get("float32_matmul_precision")) != "highest":
            raise PrefixDatasetConfigError(
                "extraction.float32_matmul_precision must be highest for publication"
            )
        if str(extraction.get("cublas_workspace_config") or "") not in {
            ":16:8",
            ":4096:8",
        }:
            raise PrefixDatasetConfigError(
                "extraction.cublas_workspace_config must select a deterministic mode"
            )
        if extraction.get("seed") is None or int(extraction["seed"]) < 0:
            raise PrefixDatasetConfigError(
                "extraction.seed must be a nonnegative integer for publication"
            )
        if bool(build.get("verify", True)):
            _required(paths, "content_family_manifest", section="paths")
            manifest_sha256 = str(
                _required(
                    dataset,
                    "content_family_manifest_sha256",
                    section="dataset",
                )
            )
            if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
                raise PrefixDatasetConfigError(
                    "dataset.content_family_manifest_sha256 must be 64 lowercase hex characters"
                )
    devices = extraction.get("devices")
    if devices is not None:
        if not isinstance(devices, (list, tuple)) or not devices:
            raise PrefixDatasetConfigError(
                "extraction.devices must be a nonempty list"
            )
        normalized_devices = [str(device).strip() for device in devices]
        if any(not device or "," in device for device in normalized_devices):
            raise PrefixDatasetConfigError(
                "extraction.devices entries must be nonempty single-device identifiers"
            )
        if len(normalized_devices) != len(set(normalized_devices)):
            raise PrefixDatasetConfigError(
                "extraction.devices must contain unique device identifiers"
            )
        extraction["devices"] = normalized_devices
    shard_alignment = str(extraction.get("shard_alignment", "document_batch"))
    if shard_alignment not in {"row", "document", "document_batch"}:
        raise PrefixDatasetConfigError(
            "extraction.shard_alignment must be row, document, or document_batch"
        )
    value_head_init = str(critic_initialization.get("value_head", "identity"))
    router_init = str(critic_initialization.get("router", "pretrained"))
    if value_head_init not in {"identity", "seeded_givens"}:
        raise PrefixDatasetConfigError(
            "critic_initialization.value_head must be identity or seeded_givens"
        )
    if router_init not in {"pretrained", "seeded_relative_noise"}:
        raise PrefixDatasetConfigError(
            "critic_initialization.router must be pretrained or seeded_relative_noise"
        )
    if (
        value_head_init == "seeded_givens"
        or router_init == "seeded_relative_noise"
    ) and critic_initialization.get("seed") is None:
        raise PrefixDatasetConfigError(
            "critic_initialization.seed is required for seeded initialization"
        )
    rotation = float(critic_initialization.get("value_head_rotation_radians", 0.2))
    router_relative_std = float(critic_initialization.get("router_relative_std", 0.01))
    if value_head_init == "seeded_givens" and rotation <= 0.0:
        raise PrefixDatasetConfigError(
            "critic_initialization.value_head_rotation_radians must be positive"
        )
    if router_init == "seeded_relative_noise" and router_relative_std <= 0.0:
        raise PrefixDatasetConfigError(
            "critic_initialization.router_relative_std must be positive"
        )
    return config


def _flag(value: Any, *, default: bool) -> str:
    return "1" if bool(default if value is None else value) else "0"


def build_launch(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    paths = config["paths"]
    dataset = config["dataset"]
    extraction = config.get("extraction") or {}
    critic_initialization = config.get("critic_initialization") or {}
    build = config.get("build") or {}
    layer = int(dataset["layer"])
    row_limit = dataset.get("row_limit")
    shard_alignment = str(extraction.get("shard_alignment", "document_batch"))
    env = {
        "CODE_ROOT": str(paths["code_root"]),
        "PY": str(config.get("python") or sys.executable),
        "MODEL": str(paths["model"]),
        "SOURCE_PARQUET": str(paths["source_parquet"]),
        "CONTRACT": str(paths["contract"]),
        "OUT": str(paths["out"]),
        "EXTRACT_ROOT": str(
            paths.get("extract_root")
            or Path(str(paths["out"])) / f"extract_prefix_{dataset['slug']}"
        ),
        "LAYER": str(layer),
        "LAYERS": str(dataset.get("layers") or f"R{layer}"),
        "ROW_START": str(int(dataset.get("row_start", 0))),
        "ROW_LIMIT": "" if row_limit is None else str(int(row_limit)),
        "SLUG": str(dataset["slug"]),
        "EXPECTED_ROWS": str(int(dataset["expected_rows"])),
        "EXPECTED_D_MODEL": str(int(dataset["expected_d_model"])),
        "EXPECTED_SOURCE_PARQUET_SHA256": str(
            dataset.get("expected_source_parquet_sha256") or ""
        ),
        "BATCH_SIZE": str(int(extraction.get("batch_size", 2))),
        "SOURCE_BATCH_SIZE": str(int(extraction.get("source_batch_size", 4096))),
        "EXTRACT_DEVICES": ",".join(
            str(device) for device in extraction.get("devices", [])
        ),
        "EXTRACT_SHARD_ALIGNMENT": shard_alignment,
        "EXTRACT_DETERMINISTIC_ALGORITHMS": _flag(
            extraction.get("deterministic_algorithms"), default=False
        ),
        "EXTRACT_ALLOW_TF32": _flag(
            extraction.get("allow_tf32"), default=False
        ),
        "EXTRACT_CUDNN_BENCHMARK": _flag(
            extraction.get("cudnn_benchmark"), default=False
        ),
        "EXTRACT_FLOAT32_MATMUL_PRECISION": str(
            extraction.get("float32_matmul_precision", "highest")
        ),
        "EXTRACT_CUBLAS_WORKSPACE_CONFIG": str(
            extraction.get("cublas_workspace_config") or ""
        ),
        "EXTRACT_SEED": (
            "" if extraction.get("seed") is None else str(int(extraction["seed"]))
        ),
        "BUILD_AR": _flag(build.get("ar"), default=True),
        "BUILD_AV": _flag(build.get("av"), default=True),
        "PREP_CRITIC": _flag(build.get("prep_critic"), default=True),
        "RUN_VERIFY": _flag(build.get("verify"), default=True),
        "PUBLICATION_MODE": _flag(
            extraction.get("publication_mode"),
            default=False,
        ),
        "MODEL_FINGERPRINT_JSON": str(paths.get("model_fingerprint_json") or ""),
        "RUNTIME_PROVENANCE_JSON": str(paths.get("runtime_provenance_json") or ""),
        "CONTENT_FAMILY_MANIFEST": str(paths.get("content_family_manifest") or ""),
        "CONTENT_FAMILY_MANIFEST_SHA256": str(
            dataset.get("content_family_manifest_sha256") or ""
        ),
        "CRITIC_VALUE_HEAD_INIT": str(
            critic_initialization.get("value_head", "identity")
        ),
        "CRITIC_INITIALIZATION_SEED": str(
            int(critic_initialization.get("seed", 0))
        ),
        "CRITIC_VALUE_HEAD_ROTATION_RADIANS": str(
            float(critic_initialization.get("value_head_rotation_radians", 0.2))
        ),
        "CRITIC_ROUTER_INIT": str(
            critic_initialization.get("router", "pretrained")
        ),
        "CRITIC_ROUTER_RELATIVE_STD": str(
            float(critic_initialization.get("router_relative_std", 0.01))
        ),
        "WANDB_MODE": "offline",
    }
    if paths.get("critic"):
        env["CRITIC"] = str(paths["critic"])
    return ["bash", "scripts/nano_prefix_dataset_pipeline.sh"], env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    command, launch_env = build_launch(config)
    payload = {
        "schema_version": "nano_prefix_dataset_launch.v1",
        "config": str(args.config),
        "command": command,
        "environment": launch_env,
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    if not args.run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    env = os.environ.copy()
    env.update(launch_env)
    return subprocess.run(
        command,
        cwd=str(config["paths"]["code_root"]),
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
