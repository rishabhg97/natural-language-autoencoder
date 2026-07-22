"""Lightweight per-step system metrics for NLA training logs.

The trainer already forwards numeric ``log_dict`` values into W&B. This module
keeps memory telemetry close to that path without adding a required dependency
or a sidecar daemon. Expensive or environment-specific probes are sampled and
best-effort: telemetry should never be able to fail a training step.
"""

from __future__ import annotations

import os
import json
import platform as platform_lib
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


GIB = 1024.0**3
MIB = 1024.0**2
PREFIX = "nla/system"
PHASE_PREFIX = "nla/phase"
ROUTER_PREFIX = "nla/router"
_PSUTIL: Any | None = None
_PSUTIL_CHECKED = False
_TORCH: Any | None = None
_TORCH_CHECKED = False
NVIDIA_SMI_FIELDS = [
    "index",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
]
NVIDIA_SMI_KEYS = [
    "index",
    "nvidia_smi_memory_used_mib",
    "nvidia_smi_memory_total_mib",
    "nvidia_smi_gpu_util_pct",
    "nvidia_smi_memory_util_pct",
    "nvidia_smi_power_w",
    "nvidia_smi_temperature_c",
]
TOPOLOGY_ENV_KEYS = {
    "NLA_WORKSPACE_GPUS": "topology_workspace_gpus",
    "NLA_ACTOR_GPUS": "topology_actor_gpus",
    "NLA_CRITIC_GPUS": "topology_critic_gpus",
    "NLA_ROLLOUT_GPUS": "topology_rollout_gpus",
    "NLA_ROLLOUT_GPUS_PER_ENGINE": "topology_rollout_gpus_per_engine",
    "NLA_SGLANG_TP_SIZE": "topology_sglang_tp_size",
    "NLA_SGLANG_BASE_GPU_ID": "topology_sglang_base_gpu_id",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _mapping_value(mapping: dict[str, Any] | None, name: str) -> str | None:
    if mapping is None or name not in mapping:
        return os.environ.get(name)
    value = mapping[name]
    return None if value is None else str(value)


def _mapping_bool(mapping: dict[str, Any] | None, name: str, default: bool = False) -> bool:
    value = _mapping_value(mapping, name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mapping_int(mapping: dict[str, Any] | None, name: str, default: int) -> int:
    value = _mapping_value(mapping, name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _topology_metrics_from_env() -> dict[str, int]:
    metrics: dict[str, int] = {}
    for env_name, metric_name in TOPOLOGY_ENV_KEYS.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed < 0:
            continue
        metrics[f"{PREFIX}/{metric_name}"] = parsed
    return metrics


def append_metrics_to_miles_loss_dict(
    log_dict: dict[str, Any],
    metrics: dict[str, float | int],
) -> dict[str, Any]:
    """Attach scalar metrics to Miles' structured per-microbatch loss dict.

    Miles loss functions return ``{"keys": [...], "values": tensor}``, where
    ``values[0]`` is the reducer normalizer. ``aggregate_train_losses`` sums
    values across microbatches/ranks and divides every metric by that same
    normalizer. Store ``metric * normalizer`` here so telemetry survives that
    reducer as an average instead of being silently ignored.
    """

    if not metrics:
        return log_dict

    keys = log_dict.get("keys")
    values = log_dict.get("values")
    torch = _load_torch()
    if torch is None:
        log_dict.update(metrics)
        return log_dict
    if not isinstance(keys, list) or not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() < 1:
        log_dict.update(metrics)
        return log_dict

    numeric_items: list[tuple[str, float]] = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            numeric_items.append((key, float(int(value))))
        elif isinstance(value, (int, float)):
            numeric_items.append((key, float(value)))
    if not numeric_items:
        return log_dict

    normalizer = values[0]
    appended = torch.as_tensor(
        [value for _key, value in numeric_items],
        dtype=values.dtype,
        device=values.device,
    ) * normalizer
    log_dict["keys"] = keys + [key for key, _value in numeric_items]
    log_dict["values"] = torch.cat([values, appended])
    return log_dict


def ru_maxrss_to_gib(value: int | float, *, platform: str | None = None) -> float:
    """Convert ``resource.ru_maxrss`` to GiB on Linux and macOS."""

    if platform is None:
        platform = sys.platform
    if platform.startswith("darwin"):
        return float(value) / GIB
    return float(value) * 1024.0 / GIB


def _process_rss_gib() -> float | None:
    try:
        return ru_maxrss_to_gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


def _load_psutil() -> Any | None:
    global _PSUTIL, _PSUTIL_CHECKED
    if not _PSUTIL_CHECKED:
        try:
            import psutil  # type: ignore[import-not-found]
        except Exception:
            psutil = None
        _PSUTIL = psutil
        _PSUTIL_CHECKED = True
    return _PSUTIL


def _load_torch() -> Any | None:
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            torch = None
        _TORCH = torch
        _TORCH_CHECKED = True
    return _TORCH


def _cuda_device_index() -> int | None:
    torch = _load_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return None


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "[N/A]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _phase_slug(value: str) -> str:
    chars: list[str] = []
    last_was_sep = False
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    slug = "".join(chars).strip("_")
    return slug or "unknown"


def _event_code(value: str | None) -> int:
    if value is None or not value.strip():
        return 0
    return 1


def parse_all_gpu_nvidia_smi_csv(text: str) -> dict[str, float | int]:
    """Parse all-GPU ``nvidia-smi`` CSV output into W&B-safe scalar keys."""

    metrics: dict[str, float | int] = {}
    memory_used_values: list[float] = []
    memory_total_values: list[float] = []
    util_values: list[float] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        values = [_parse_float(part) for part in line.split(",")]
        if len(values) != len(NVIDIA_SMI_KEYS) or values[0] is None:
            continue
        gpu_index = int(values[0])
        gpu_prefix = f"{PREFIX}/gpu{gpu_index}"
        for key, value in zip(NVIDIA_SMI_KEYS[1:], values[1:]):
            if value is None:
                continue
            metrics[f"{gpu_prefix}/{key}"] = value
        memory_used = values[1]
        memory_total = values[2]
        util = values[3]
        if memory_used is not None:
            memory_used_values.append(memory_used)
        if memory_total is not None:
            memory_total_values.append(memory_total)
        if util is not None:
            util_values.append(util)

    if memory_used_values:
        metrics[f"{PREFIX}/all_gpu_count"] = len(memory_used_values)
        metrics[f"{PREFIX}/all_gpu_memory_used_mib"] = float(sum(memory_used_values))
        metrics[f"{PREFIX}/all_gpu_memory_used_mib_max"] = float(max(memory_used_values))
    if memory_total_values:
        metrics[f"{PREFIX}/all_gpu_memory_total_mib"] = float(sum(memory_total_values))
    if util_values:
        metrics[f"{PREFIX}/all_gpu_util_pct_mean"] = float(sum(util_values) / len(util_values))
        metrics[f"{PREFIX}/all_gpu_util_pct_max"] = float(max(util_values))
    return metrics


class RouterEntropyTracker:
    """Best-effort router-load telemetry for MoE checkpoints.

    Hooks look for integer top-k expert-index tensors in router outputs. If the
    remote code shape changes, the tracker simply emits nothing.
    """

    def __init__(self, expert_count: int | None = None, torch_module: Any | None = None):
        self.expert_count = expert_count
        self._torch = torch_module if torch_module is not None else _load_torch()
        self._counts: Any | None = None
        self._counts_by_layer: dict[str, Any] = {}
        self._expert_count_by_layer: dict[str, int] = {}
        self._handles: list[Any] = []

    @staticmethod
    def _module_expert_count(module: Any) -> int | None:
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        experts = getattr(module, "experts", None)
        if experts is not None:
            try:
                return len(experts)
            except Exception:
                return None
        return None

    @classmethod
    def _find_index_tensor(cls, value: Any, torch_module: Any | None = None) -> Any | None:
        torch = torch_module if torch_module is not None else _load_torch()
        if torch is None:
            return None
        if isinstance(value, torch.Tensor) and value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = cls._find_index_tensor(item, torch)
                if found is not None:
                    return found
        if isinstance(value, dict):
            for item in value.values():
                found = cls._find_index_tensor(item, torch)
                if found is not None:
                    return found
        return None

    @classmethod
    def attach(cls, model: Any) -> "RouterEntropyTracker":
        tracker = cls()
        if tracker._torch is None:
            return tracker
        for name, module in model.named_modules():
            class_name = type(module).__name__.lower()
            if "router" not in class_name and "gate" not in class_name:
                continue
            expert_count = cls._module_expert_count(module)
            if expert_count is not None:
                tracker.expert_count = max(tracker.expert_count or 0, expert_count)
                tracker._expert_count_by_layer[name or "root"] = expert_count
            tracker._handles.append(
                module.register_forward_hook(
                    lambda hooked, inputs, output, layer_name=name or "root": tracker._hook(
                        hooked,
                        inputs,
                        output,
                        layer_name=layer_name,
                    )
                )
            )
        return tracker

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(
        self,
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
        *,
        layer_name: str | None = None,
    ) -> None:
        torch = self._torch
        if torch is None:
            return
        indices = self._find_index_tensor(output, torch)
        if indices is None:
            return
        expert_count = self.expert_count
        if expert_count is None:
            max_idx = int(indices.detach().max().item()) if indices.numel() else -1
            expert_count = max_idx + 1
            self.expert_count = expert_count
        if expert_count <= 0:
            return
        counts = torch.bincount(indices.detach().reshape(-1).to(torch.long).cpu(), minlength=expert_count)
        self._counts = counts if self._counts is None else self._counts + counts
        if layer_name is not None:
            layer_expert_count = self._expert_count_by_layer.get(
                layer_name, expert_count
            )
            layer_counts = torch.bincount(
                indices.detach().reshape(-1).to(torch.long).cpu(),
                minlength=layer_expert_count,
            )
            previous = self._counts_by_layer.get(layer_name)
            self._counts_by_layer[layer_name] = (
                layer_counts if previous is None else previous + layer_counts
            )

    @staticmethod
    def _metrics_for_counts(prefix: str, counts: Any, torch: Any) -> dict[str, float | int]:
        total = counts.sum().float()
        probs = counts.float() / total.clamp_min(1.0)
        nonzero = probs > 0
        entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
        max_entropy = torch.log(torch.tensor(float(len(counts)))).clamp_min(1e-12)
        active = int((counts > 0).sum().item())
        return {
            f"{prefix}/expert_count": int(len(counts)),
            f"{prefix}/active_expert_count": active,
            f"{prefix}/token_assignments": int(total.item()),
            f"{prefix}/router_entropy": float(entropy.item()),
            f"{prefix}/router_entropy_normalized": float((entropy / max_entropy).item()),
            f"{prefix}/max_expert_fraction": float((counts.max().float() / total).item()),
            f"{prefix}/min_active_expert_fraction": float((counts[counts > 0].min().float() / total).item()),
        }

    def collect(self) -> dict[str, float | int]:
        torch = self._torch
        if torch is None:
            return {}
        counts = self._counts
        self._counts = None
        counts_by_layer = self._counts_by_layer
        self._counts_by_layer = {}
        if counts is None or counts.sum().item() <= 0:
            return {}
        metrics = self._metrics_for_counts(ROUTER_PREFIX, counts, torch)
        for layer_name, layer_counts in sorted(counts_by_layer.items()):
            if layer_counts.sum().item() <= 0:
                continue
            layer_key = layer_name.replace(".", "/")
            metrics.update(
                self._metrics_for_counts(
                    f"{ROUTER_PREFIX}/layers/{layer_key}",
                    layer_counts,
                    torch,
                )
            )
        return metrics


@dataclass
class SystemMetricsLogger:
    enabled: bool = False
    interval_steps: int = 1
    nvidia_smi_interval_steps: int = 0
    rank: int = 0
    local_rank: int = 0
    role: str = "actor"
    include_nvidia_smi: bool = True
    phase_metrics_enabled: bool = True
    phase_metrics_all_gpus: bool = True
    phase_metrics_wandb: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        rank: int = 0,
        local_rank: int = 0,
        role: str = "actor",
    ) -> "SystemMetricsLogger":
        nvidia_smi_interval_steps = _env_int("NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS", 0)
        enabled = _env_bool("NLA_SYSTEM_METRICS", False)
        return cls(
            enabled=enabled,
            interval_steps=max(1, _env_int("NLA_SYSTEM_METRICS_INTERVAL_STEPS", 1)),
            nvidia_smi_interval_steps=nvidia_smi_interval_steps,
            rank=rank,
            local_rank=local_rank,
            role=role,
            include_nvidia_smi=nvidia_smi_interval_steps > 0,
            phase_metrics_enabled=_env_bool("NLA_PHASE_METRICS", enabled),
            phase_metrics_all_gpus=_env_bool("NLA_PHASE_METRICS_ALL_GPUS", True),
            phase_metrics_wandb=_env_bool("NLA_PHASE_METRICS_WANDB", True),
        )

    @classmethod
    def from_env_mapping(
        cls,
        mapping: dict[str, Any] | None,
        *,
        rank: int = 0,
        local_rank: int = 0,
        role: str = "actor",
    ) -> "SystemMetricsLogger":
        nvidia_smi_interval_steps = _mapping_int(mapping, "NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS", 0)
        enabled = _mapping_bool(mapping, "NLA_SYSTEM_METRICS", False)
        return cls(
            enabled=enabled,
            interval_steps=max(1, _mapping_int(mapping, "NLA_SYSTEM_METRICS_INTERVAL_STEPS", 1)),
            nvidia_smi_interval_steps=nvidia_smi_interval_steps,
            rank=rank,
            local_rank=local_rank,
            role=role,
            include_nvidia_smi=nvidia_smi_interval_steps > 0,
            phase_metrics_enabled=_mapping_bool(mapping, "NLA_PHASE_METRICS", enabled),
            phase_metrics_all_gpus=_mapping_bool(mapping, "NLA_PHASE_METRICS_ALL_GPUS", True),
            phase_metrics_wandb=_mapping_bool(mapping, "NLA_PHASE_METRICS_WANDB", True),
        )

    def should_collect(self, step_id: int) -> bool:
        return self.enabled and step_id % max(1, self.interval_steps) == 0

    def collect(self, *, step_id: int) -> dict[str, float | int]:
        if not self.should_collect(step_id):
            return {}
        return self._collect_now(step_id=step_id, include_nvidia_smi=self.include_nvidia_smi)

    def _collect_now(self, *, step_id: int, include_nvidia_smi: bool) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            f"{PREFIX}/rank": int(self.rank),
            f"{PREFIX}/local_rank": int(self.local_rank),
            f"{PREFIX}/pid": int(os.getpid()),
            f"{PREFIX}/timestamp_unix": float(time.time()),
        }
        rss_gib = _process_rss_gib()
        if rss_gib is not None:
            metrics[f"{PREFIX}/process_maxrss_gib"] = rss_gib

        metrics.update(self._psutil_metrics())
        metrics.update(self._cuda_memory_metrics())
        metrics.update(_topology_metrics_from_env())
        if (
            include_nvidia_smi
            and self.nvidia_smi_interval_steps > 0
            and step_id % self.nvidia_smi_interval_steps == 0
        ):
            metrics.update(self._nvidia_smi_metrics())
        return metrics

    def collect_phase(
        self,
        *,
        step_id: int,
        phase: str,
        event: str | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = True,
    ) -> dict[str, float | int]:
        if not self.enabled or not self.phase_metrics_enabled:
            return {}
        base = (
            self._collect_now(step_id=step_id, include_nvidia_smi=True)
            if force
            else self.collect(step_id=step_id)
        )
        if self.phase_metrics_all_gpus:
            base.update(self._all_gpu_nvidia_smi_metrics())
        phase_prefix = f"{PHASE_PREFIX}/{_phase_slug(phase)}"
        metrics: dict[str, float | int] = {
            f"{phase_prefix}/event_code": _event_code(event),
            f"{phase_prefix}/snapshot": 1,
        }
        for key, value in base.items():
            if not isinstance(value, (int, float, bool)):
                continue
            suffix = key[len(PREFIX) + 1 :] if key.startswith(f"{PREFIX}/") else key
            metrics[f"{phase_prefix}/{suffix}"] = float(int(value)) if isinstance(value, bool) else value
        for key, value in (extra or {}).items():
            if isinstance(value, bool):
                metrics[f"{phase_prefix}/{_phase_slug(str(key))}"] = int(value)
            elif isinstance(value, (int, float)):
                metrics[f"{phase_prefix}/{_phase_slug(str(key))}"] = value
        return metrics

    def emit_phase_snapshot(
        self,
        *,
        step_id: int,
        phase: str,
        event: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float | int]:
        metrics = self.collect_phase(step_id=step_id, phase=phase, event=event, extra=extra)
        if not metrics:
            return {}
        payload = {
            "schema_version": "nla_observability_phase.v1",
            "role": self.role,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "phase": phase,
            "event": event,
            "step_id": step_id,
            "extra": extra or {},
            "metrics": metrics,
        }
        try:
            print("[NLA OBS] " + json.dumps(payload, sort_keys=True), flush=True)
        except Exception:
            pass
        if self.phase_metrics_wandb:
            try:
                import wandb  # type: ignore[import-not-found]

                if getattr(wandb, "run", None) is not None:
                    wandb.log(metrics, step=step_id)
            except Exception:
                pass
        return metrics

    def _psutil_metrics(self) -> dict[str, float | int]:
        psutil = _load_psutil()
        if psutil is None:
            return {}
        try:
            process = psutil.Process(os.getpid())
            memory = process.memory_info()
            virtual_memory = psutil.virtual_memory()
            metrics: dict[str, float | int] = {
                f"{PREFIX}/process_rss_gib": float(memory.rss) / GIB,
                f"{PREFIX}/process_vms_gib": float(memory.vms) / GIB,
                f"{PREFIX}/process_cpu_pct": float(process.cpu_percent(interval=None)),
                f"{PREFIX}/process_num_threads": int(process.num_threads()),
                f"{PREFIX}/system_ram_available_gib": float(virtual_memory.available) / GIB,
                f"{PREFIX}/system_ram_used_pct": float(virtual_memory.percent),
            }
            try:
                io = process.io_counters()
            except Exception:
                io = None
            if io is not None:
                metrics[f"{PREFIX}/process_read_gib"] = float(io.read_bytes) / GIB
                metrics[f"{PREFIX}/process_write_gib"] = float(io.write_bytes) / GIB
            return metrics
        except Exception:
            return {}

    def _cuda_memory_metrics(self) -> dict[str, float | int]:
        torch = _load_torch()
        if torch is None or not torch.cuda.is_available():
            return {}
        try:
            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception:
            free_bytes = None
            total_bytes = None
        metrics: dict[str, float | int] = {
            f"{PREFIX}/cuda_memory_allocated_gib": torch.cuda.memory_allocated() / GIB,
            f"{PREFIX}/cuda_memory_reserved_gib": torch.cuda.memory_reserved() / GIB,
            f"{PREFIX}/cuda_max_memory_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            f"{PREFIX}/cuda_max_memory_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        }
        if free_bytes is not None and total_bytes is not None:
            metrics[f"{PREFIX}/cuda_free_gib"] = float(free_bytes) / GIB
            metrics[f"{PREFIX}/cuda_total_gib"] = float(total_bytes) / GIB
        return metrics

    def _nvidia_smi_metrics(self) -> dict[str, float | int]:
        device = _cuda_device_index()
        if device is None:
            return {}
        fields = [
            "memory.used",
            "memory.total",
            "utilization.gpu",
            "utilization.memory",
            "power.draw",
            "temperature.gpu",
        ]
        command = [
            "nvidia-smi",
            "--id",
            str(device),
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return {}
        line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
        values = [_parse_float(part) for part in line.split(",")]
        if len(values) != len(fields):
            return {}
        keys = [
            "nvidia_smi_memory_used_mib",
            "nvidia_smi_memory_total_mib",
            "nvidia_smi_gpu_util_pct",
            "nvidia_smi_memory_util_pct",
            "nvidia_smi_power_w",
            "nvidia_smi_temperature_c",
        ]
        return {
            f"{PREFIX}/{key}": value
            for key, value in zip(keys, values)
            if value is not None
        }

    def _all_gpu_nvidia_smi_metrics(self) -> dict[str, float | int]:
        command = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return {}
        return parse_all_gpu_nvidia_smi_csv(completed.stdout)


def runtime_summary() -> dict[str, str | int | bool]:
    """Small diagnostic summary for run manifests and tests."""

    torch = _load_torch()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    return {
        "system_metrics_enabled": _env_bool("NLA_SYSTEM_METRICS", False),
        "python": platform_lib.python_version(),
        "platform": platform_lib.platform(),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
