from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_ar_readout_diagnostic.py"
    spec = importlib.util.spec_from_file_location("nano_ar_readout_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ridge_with_bias_recovers_linear_mapping():
    diag = load_script()
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.concatenate([2.0 * x + 3.0, -1.0 * x + 4.0], axis=1)

    model = diag.fit_ridge(x, y, ridge=1e-8, bias=True)
    pred = diag.predict_ridge(x, model)

    np.testing.assert_allclose(pred, y, atol=1e-4)


def test_row_rms_normalize_keeps_zero_rows_finite():
    diag = load_script()
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    out = diag.row_rms_normalize(x)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], np.array([0.84852814, 1.1313709], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out[1], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)
