import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "verify_nano_av_eval_report.py"
    spec = importlib.util.spec_from_file_location("verify_nano_av_eval_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(real=0.8, shuffled=1.3, test_count=0):
    controls = {
        "real": real,
        "shuffled": shuffled,
        "zero": 1.2,
        "mean": 1.25,
        "none": 1.22,
    }
    return {
        "eval_splits": ["validation"],
        "loss_summary": {
            control: {
                "validation": {"count": 512, "loss": loss},
                "test": {"count": test_count, "loss": None},
            }
            for control, loss in controls.items()
        },
    }


def config(report_path):
    return {
        "report_json": str(report_path),
        "expected": {
            "eval_splits": ["validation"],
            "counts": {"validation": 512},
            "controls": ["real", "shuffled", "zero", "mean", "none"],
            "min_real_control_gap": 0.0,
            "forbid_unrequested_rows": True,
        },
    }


def test_verifier_accepts_finite_validation_only_control_win(tmp_path):
    module = load_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report()))

    result = module.verify(config(report_path))

    assert result["passed"]
    assert result["split_results"]["validation"]["controls"]["shuffled"][
        "control_minus_real"
    ] == 0.5


def test_verifier_rejects_control_loss_and_test_consumption(tmp_path):
    module = load_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report(real=1.4, shuffled=1.3, test_count=4)))

    result = module.verify(config(report_path))

    assert not result["passed"]
    assert any("does not beat shuffled" in error for error in result["errors"])
    assert any("unrequested split test" in error for error in result["errors"])
