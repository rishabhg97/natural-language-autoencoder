from __future__ import annotations

import json
from pathlib import Path

import pytest

from observatory.launch_after_prepare import process_exists, validate_preparation


def test_validate_preparation_requires_converted_hf(tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors").write_bytes(b"weights")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"passed": True, "av_stage": {"status": "converted"}}))
    assert validate_preparation(report, hf)["passed"] is True


def test_validate_preparation_rejects_skipped_stage(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"passed": True, "av_stage": {"status": "skipped"}}))
    with pytest.raises(ValueError, match="did not complete"):
        validate_preparation(report, tmp_path / "hf")


def test_process_exists_reports_current_process() -> None:
    import os

    assert process_exists(os.getpid())
