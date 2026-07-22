import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_nano_training_metrics.py"


def load_script():
    spec = importlib.util.spec_from_file_location("export_nano_training_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parses_step_and_perf_curves(tmp_path):
    module = load_script()
    path = tmp_path / "train.log"
    path.write_text(
        "\x1b[36mworker\x1b[0m step 0: {'train/loss': 2.0, 'train/pid': 7}\n"
        "worker perf 0: {'perf/actor_train_time': 3.0}\n"
        "worker step 1: {'train/loss': 1.0, 'train/pid': 8}\n"
    )

    rows = module.parse_curves(
        path,
        expected_updates=2,
        include_prefixes=("train/", "perf/"),
        exclude_metrics={"train/pid"},
    )

    assert rows == [
        {"step": 0, "perf/actor_train_time": 3.0, "train/loss": 2.0},
        {"step": 1, "train/loss": 1.0},
    ]


def test_rejects_missing_step(tmp_path):
    module = load_script()
    path = tmp_path / "train.log"
    path.write_text("worker step 1: {'train/loss': 1.0}\n")

    with pytest.raises(module.MetricExportError, match="steps do not match"):
        module.parse_curves(
            path,
            expected_updates=2,
            include_prefixes=("train/",),
            exclude_metrics=set(),
        )


def test_rejects_nonfinite_metric(tmp_path):
    module = load_script()
    path = tmp_path / "train.log"
    path.write_text("worker step 0: {'train/loss': float('nan')}\n")

    with pytest.raises(module.MetricExportError, match="could not parse"):
        module.parse_curves(
            path,
            expected_updates=1,
            include_prefixes=("train/",),
            exclude_metrics=set(),
        )
