from pathlib import Path

import pytest

from scripts.build_nano_compute_accounting import AccountingError, build_report


def _write_log(path: Path, *, steps: int = 2) -> None:
    lines = [
        "# started_utc=2026-07-16T00:00:00Z",
        "/venv/python /code/train.py --global-batch-size 192 "
        "--micro-batch-size 48 --lr 5e-5 --num-rollout 2 "
        "--actor-num-gpus-per-node 4",
    ]
    for step in range(steps):
        payload = {
            "train/loss": 1.0 - 0.1 * step,
            "train/step": step,
            "train/nla/system/cuda_max_memory_allocated_gib": 20.0 + step,
            "train/nla/system/nvidia_smi_gpu_util_pct": 80.0 + step,
        }
        lines.append(f"[actor] step {step}: {payload!r}")
    lines.append("# completed_utc=2026-07-16T00:30:00Z")
    path.write_text("\n".join(lines) + "\n")


def _config(log: Path) -> dict:
    return {
        "schema_version": "nano_compute_accounting.v1",
        "runs": [
            {
                "name": "test-run",
                "component": "AR",
                "train_log": str(log),
                "gpu_type": "H100",
                "gpu_count": 4,
                "expected_updates": 2,
            }
        ],
    }


def test_build_report_accounts_wall_time_steps_and_envelope(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    _write_log(log)
    report = build_report(_config(log))

    assert report["successful_training_gpu_hours"] == pytest.approx(2.0)
    run = report["runs"][0]
    assert run["optimizer_updates"] == 2
    assert run["logged_optimizer_steps"] == 2
    assert run["resolved_command"]["global_batch_size"] == "192"
    assert run["system_envelope"]["cuda_max_memory_allocated_gib_max"] == 21.0
    assert len(run["train_log_sha256"]) == 64


def test_build_report_fails_on_missing_optimizer_step(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    _write_log(log, steps=1)

    with pytest.raises(AccountingError, match="optimizer steps do not match"):
        build_report(_config(log))
