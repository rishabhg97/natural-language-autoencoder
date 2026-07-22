"""Optional W&B tracking helpers for Nano NLA experiments.

The helpers deliberately degrade to a no-op when disabled, missing, or unable
to initialize. A training run should not fail only because telemetry is down.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = "nano30b-nla-pilot"


def parse_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def load_env_file(path: str | Path | None) -> bool:
    """Load KEY=VALUE pairs from a dotenv-style file without overwriting env."""

    if path is None:
        return False
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return False
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value
    for alias in ("WANB_API_KEY", "wandb_api_key"):
        if "WANDB_API_KEY" not in os.environ and alias in os.environ:
            os.environ["WANDB_API_KEY"] = os.environ[alias]
    return True


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=parse_env_bool("NANO_WANDB", True),
        help="Enable optional Weights & Biases logging. Set NANO_WANDB=0 or pass --no-wandb to disable.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument(
        "--wandb-env-file",
        type=Path,
        default=Path(os.environ.get("NANO_ENV_FILE", ".env")),
        help="Dotenv file to source W&B env vars from. Missing files are ignored.",
    )


def _tags_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _is_loggable_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float | int | bool]:
    """Flatten nested numeric/bool dict values, skipping lists and text payloads."""

    flat: dict[str, float | int | bool] = {}
    for key, value in payload.items():
        key_text = str(key).replace(".", "_")
        name = f"{prefix}/{key_text}" if prefix else key_text
        if isinstance(value, dict):
            flat.update(flatten_numeric(value, name))
        elif _is_loggable_scalar(value):
            flat[name] = value
    return flat


class WandbTracker:
    def __init__(self, run: Any | None, metadata: dict[str, Any]):
        self.run = run
        self.metadata = metadata

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        clean = {key: value for key, value in metrics.items() if _is_loggable_scalar(value)}
        if clean:
            self.run.log(clean, step=step)

    def log_history(self, history: list[dict[str, Any]] | None, prefix: str) -> None:
        if not history:
            return
        for item in history:
            step = item.get("step")
            step_int = int(step) if isinstance(step, int) else None
            metrics = {
                f"{prefix}/{key}": value
                for key, value in item.items()
                if key != "step" and _is_loggable_scalar(value)
            }
            self.log(metrics, step=step_int)

    def log_summary(self, payload: dict[str, Any], prefix: str = "") -> None:
        self.log(flatten_numeric(payload, prefix=prefix))

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if self.run is None:
            return
        if summary:
            for key, value in summary.items():
                if _is_loggable_scalar(value):
                    self.run.summary[key] = value
        self.run.finish()


def init_wandb(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    job_type: str,
    config: dict[str, Any],
) -> WandbTracker:
    """Create a W&B run or a no-op tracker.

    Secrets are read only through environment variables or dotenv and are never
    echoed into metadata/config.
    """

    metadata: dict[str, Any] = {
        "enabled": bool(getattr(args, "wandb", False)),
        "status": "disabled",
        "project": getattr(args, "wandb_project", DEFAULT_PROJECT),
    }
    load_env_file(getattr(args, "wandb_env_file", None))
    if not getattr(args, "wandb", False):
        return WandbTracker(None, metadata)

    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when dep absent in real envs
        metadata.update({"status": "import_failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[nano_wandb] disabled: failed to import wandb ({type(exc).__name__})", file=sys.stderr)
        return WandbTracker(None, metadata)

    init_kwargs: dict[str, Any] = {
        "project": getattr(args, "wandb_project", DEFAULT_PROJECT),
        "entity": getattr(args, "wandb_entity", None),
        "name": getattr(args, "wandb_name", None) or run_dir.name,
        "group": getattr(args, "wandb_group", None),
        "tags": _tags_from_text(getattr(args, "wandb_tags", "")),
        "job_type": job_type,
        "dir": str(run_dir),
        "config": config,
    }
    mode = getattr(args, "wandb_mode", None)
    if mode:
        init_kwargs["mode"] = mode
    init_kwargs = {key: value for key, value in init_kwargs.items() if value not in (None, [], "")}

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - network/auth failures are env-specific
        metadata.update({"status": "init_failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[nano_wandb] disabled: wandb.init failed ({type(exc).__name__})", file=sys.stderr)
        return WandbTracker(None, metadata)

    metadata.update(
        {
            "status": "enabled",
            "project": init_kwargs.get("project"),
            "name": init_kwargs.get("name"),
            "group": init_kwargs.get("group"),
            "mode": init_kwargs.get("mode"),
        }
    )
    return WandbTracker(run, metadata)
