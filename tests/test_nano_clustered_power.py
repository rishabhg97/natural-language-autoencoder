import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "nano_clustered_power.py"
    spec = importlib.util.spec_from_file_location("nano_clustered_power", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_report(path, losses, families):
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "validation": {
                        "row_keys": [f"row-{index}" for index in range(len(losses))],
                        "content_family_ids": families,
                        "rowwise_directional_mse": {"av_real": losses},
                    }
                }
            }
        )
    )


def test_power_report_uses_paired_family_residuals(tmp_path):
    module = load_module()
    families = [f"family-{index:03d}" for index in range(200)]
    baseline_losses = np.full(200, 0.20).tolist()
    candidate_losses = (0.18 + np.linspace(-0.002, 0.002, 200)).tolist()
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    write_report(candidate, candidate_losses, families)
    write_report(baseline, baseline_losses, families)

    report = module.build_report(
        candidate,
        baseline,
        split="validation",
        variant="av_real",
        target_relative_gain=0.10,
        registered_family_count=200,
        simulations=1_000,
        seed=7,
        min_power=0.80,
    )

    assert report["passed"]
    assert report["pilot_family_count"] == 200
    assert report["row_count"] == 200
    assert report["target_absolute_gain"] == pytest.approx(0.02)
    assert report["power"] >= 0.80


def test_power_report_rejects_mismatched_rows(tmp_path):
    module = load_module()
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    write_report(candidate, [0.1, 0.2], ["a", "b"])
    write_report(baseline, [0.2], ["a"])

    with pytest.raises(module.ClusteredPowerError, match="row identities differ"):
        module.build_report(
            candidate,
            baseline,
            split="validation",
            variant="av_real",
            target_relative_gain=0.10,
            registered_family_count=None,
            simulations=100,
            seed=7,
            min_power=0.80,
        )
