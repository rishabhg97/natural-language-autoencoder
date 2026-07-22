#!/usr/bin/env python3
"""Manage a config-driven local OpenAI-compatible inference server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nano_source_provenance  # noqa: E402


SCHEMA_VERSION = "nano_openai_compatible_server.v1"


class ServerError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ServerError(f"config must use schema_version {SCHEMA_VERSION}")
    for section in ("server", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ServerError(f"config requires mapping: {section}")
    for key in ("python", "module", "model_path", "served_model_name"):
        if not config["server"].get(key):
            raise ServerError(f"server.{key} is required")
    for key in ("pid_file", "log_file"):
        if not config["runtime"].get(key):
            raise ServerError(f"runtime.{key} is required")
    stage = config["runtime"].get("model_stage")
    if stage is not None:
        if not isinstance(stage, dict):
            raise ServerError("runtime.model_stage must be a mapping")
        for key in ("output_dir", "manifest_json"):
            if not stage.get(key):
                raise ServerError(f"runtime.model_stage.{key} is required")
        if int(stage.get("workers", 4)) < 1:
            raise ServerError("runtime.model_stage.workers must be at least 1")
        if int(stage.get("task_size_bytes", 512 * 1024 * 1024)) < 1:
            raise ServerError(
                "runtime.model_stage.task_size_bytes must be at least 1"
            )
    return config


def effective_model_path(config: dict[str, Any]) -> Path:
    stage = config["runtime"].get("model_stage") or {}
    return Path(stage.get("output_dir") or config["server"]["model_path"])


def prepare_model_stage(config: dict[str, Any]) -> dict[str, Any] | None:
    stage = config["runtime"].get("model_stage")
    if not stage:
        return None
    source = Path(config["server"]["model_path"]).resolve()
    destination = Path(stage["output_dir"])
    manifest_path = Path(stage["manifest_json"])
    follow_symlinks = bool(stage.get("follow_symlinks", False))
    source_signature = nano_source_provenance.directory_stat_signature(
        source,
        follow_symlinks=follow_symlinks,
    )
    if destination.exists():
        if not manifest_path.is_file():
            raise ServerError(
                f"staged model exists without provenance manifest: {destination}"
            )
        report = json.loads(manifest_path.read_text())
        destination_signature = nano_source_provenance.directory_stat_signature(
            destination
        )
        expected = {
            "source_dir": str(source),
            "source_stat_signature": source_signature,
            "destination_stat_signature": destination_signature,
        }
        mismatches = {
            key: {"manifest": report.get(key), "current": value}
            for key, value in expected.items()
            if report.get(key) != value
        }
        if mismatches:
            raise ServerError(
                "staged model provenance mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return {**report, "reused": True}

    fingerprint = nano_source_provenance.fingerprint_and_copy_directory(
        source,
        destination,
        label="openai_compatible_server_model",
        workers=int(stage.get("workers", 4)),
        task_size=int(stage.get("task_size_bytes", 512 * 1024 * 1024)),
        follow_symlinks=follow_symlinks,
    )
    report = {
        "schema_version": "nano_openai_compatible_server_model_stage.v1",
        "source_dir": str(source),
        "destination_dir": str(destination),
        "source_stat_signature": source_signature,
        "destination_stat_signature": (
            nano_source_provenance.directory_stat_signature(destination)
        ),
        "fingerprint": fingerprint,
        "workers": int(stage.get("workers", 4)),
        "task_size_bytes": int(
            stage.get("task_size_bytes", 512 * 1024 * 1024)
        ),
        "reused": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return report


def build_command(config: dict[str, Any]) -> list[str]:
    server = config["server"]
    command = [
        str(server["python"]),
        "-m",
        str(server["module"]),
        "--model-path",
        str(effective_model_path(config)),
        "--served-model-name",
        str(server["served_model_name"]),
        "--host",
        str(server.get("host", "127.0.0.1")),
        "--port",
        str(int(server.get("port", 30080))),
        "--tp-size",
        str(int(server.get("tp_size", 1))),
        "--mem-fraction-static",
        str(float(server.get("mem_fraction_static", 0.88))),
        "--context-length",
        str(int(server.get("context_length", 4096))),
    ]
    if bool(server.get("trust_remote_code", False)):
        command.append("--trust-remote-code")
    command.extend(str(value) for value in server.get("extra_args", []))
    return command


def endpoint(config: dict[str, Any]) -> str:
    server = config["server"]
    return f"http://{server.get('host', '127.0.0.1')}:{int(server.get('port', 30080))}"


def _pid(config: dict[str, Any]) -> int | None:
    path = Path(config["runtime"]["pid_file"])
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _ready(config: dict[str, Any]) -> bool:
    try:
        with urllib.request.urlopen(endpoint(config) + "/v1/models", timeout=2) as response:
            value = json.loads(response.read())
        return response.status == 200 and isinstance(value.get("data"), list)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def status(config: dict[str, Any]) -> dict[str, Any]:
    pid = _pid(config)
    stage = config["runtime"].get("model_stage") or {}
    manifest_path = Path(stage["manifest_json"]) if stage else None
    return {
        "schema_version": SCHEMA_VERSION,
        "pid": pid,
        "process_alive": _alive(pid),
        "endpoint": endpoint(config),
        "ready": _ready(config),
        "log_file": str(config["runtime"]["log_file"]),
        "model_path": str(effective_model_path(config)),
        "model_stage_manifest": (
            str(manifest_path) if manifest_path and manifest_path.is_file() else None
        ),
    }


def start(config: dict[str, Any]) -> dict[str, Any]:
    current = status(config)
    if current["process_alive"] and current["ready"]:
        return {**current, "reused": True}
    if current["process_alive"]:
        raise ServerError(
            f"server PID {current['pid']} is alive but endpoint is not ready"
        )
    stage_report = prepare_model_stage(config)
    runtime = config["runtime"]
    pid_file = Path(runtime["pid_file"])
    log_file = Path(runtime["log_file"])
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in runtime.get("environment", {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = str(config["server"].get("gpu_device", "0"))
    with log_file.open("a") as log:
        process = subprocess.Popen(
            build_command(config),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{process.pid}\n")
    deadline = time.monotonic() + int(runtime.get("startup_timeout_seconds", 900))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ServerError(
                f"server exited during startup with return code {process.returncode}; "
                f"see {log_file}"
            )
        if _ready(config):
            return {
                **status(config),
                "reused": False,
                "model_stage": stage_report,
            }
        time.sleep(2)
    raise ServerError(f"server did not become ready before timeout; see {log_file}")


def stop(config: dict[str, Any]) -> dict[str, Any]:
    pid = _pid(config)
    if not _alive(pid):
        return {**status(config), "stopped": False}
    assert pid is not None
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    return {**status(config), "stopped": not _alive(pid)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "start", "status", "stop"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "render":
        result = {"command": build_command(config), "endpoint": endpoint(config)}
    elif args.command == "start":
        result = start(config)
    elif args.command == "status":
        result = status(config)
    else:
        result = stop(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
