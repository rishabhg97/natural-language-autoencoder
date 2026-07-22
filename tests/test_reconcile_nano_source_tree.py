from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reconcile_removes_only_unlisted_source_files(tmp_path: Path) -> None:
    provenance = load_script("nano_source_provenance")
    reconcile = load_script("reconcile_nano_source_tree")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    expected = scripts / "expected.py"
    expected.write_text("pass\n")
    manifest = provenance.source_file_manifest(tmp_path, roots=("scripts",))
    extra = scripts / "stale.py"
    extra.write_text("stale\n")

    report = reconcile.reconcile_source_tree(
        code_root=tmp_path,
        manifest=manifest,
        apply=True,
    )

    assert report["matched"] is True
    assert report["deleted_files"] == ["scripts/stale.py"]
    assert expected.is_file()
    assert not extra.exists()
