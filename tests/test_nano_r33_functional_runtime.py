from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from nano_r33_functional_runtime import prepare_local_target_remote_code  # noqa: E402


def test_prepare_local_target_remote_code_uses_canonical_patcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    package = types.ModuleType("nla")
    package.__path__ = []  # type: ignore[attr-defined]
    patches = types.ModuleType("nla.remote_code_patches")

    def prepare(path: Path) -> object:
        calls.append(Path(path))
        return object()

    patches.prepare_nemotron_h_checkpoint_for_load = prepare  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nla", package)
    monkeypatch.setitem(sys.modules, "nla.remote_code_patches", patches)

    assert prepare_local_target_remote_code(
        tmp_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    assert calls == [tmp_path]


def test_prepare_local_target_remote_code_skips_nonlocal_models(tmp_path: Path) -> None:
    assert not prepare_local_target_remote_code(
        tmp_path,
        trust_remote_code=True,
        local_files_only=False,
    )
