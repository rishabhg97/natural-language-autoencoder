from __future__ import annotations

import json
from pathlib import Path

import pytest

from observatory.pipeline_supervisor import (
    validate_critic_preparation,
    write_state,
)


def _checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


def test_validate_critic_preparation_requires_two_complete_stages(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    independent = tmp_path / "independent"
    _checkpoint(primary)
    _checkpoint(independent)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "passed": True,
                "critic_stages": {
                    "primary_ar": {"status": "copied"},
                    "independent_ar": {"status": "reused"},
                },
            }
        )
    )
    parsed = validate_critic_preparation(report, [primary, independent])
    assert parsed["passed"] is True


def test_validate_critic_preparation_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "passed": True,
                "critic_stages": {
                    "primary_ar": {"status": "copied"},
                    "independent_ar": {"status": "copied"},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_critic_preparation(report, [tmp_path / "missing"])


def test_write_state_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_state(path, {"status": "running"})
    assert json.loads(path.read_text()) == {"status": "running"}
