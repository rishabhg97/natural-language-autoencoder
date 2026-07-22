import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_nano_public_release_bundle.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "package_nano_public_release_bundle", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def config_for(tmp_path: Path, module) -> dict:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "README.md").write_text("candidate\n")
    entries = module._tree_entries(root)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "automatic_gate_passed": True,
                "root": str(root.resolve()),
                "tree_manifest_sha256": module._tree_hash(entries),
            }
        )
    )
    return {
        "schema_version": module.SCHEMA_VERSION,
        "release_id": "test",
        "bundle_root": str(root),
        "audit_report": str(audit),
        "output_archive": str(tmp_path / "candidate.tgz"),
        "attestation_json": str(tmp_path / "attestation.json"),
        "archive_root_name": "candidate",
    }


def test_packages_exact_audited_tree_deterministically(tmp_path):
    module = load_script()
    config = config_for(tmp_path, module)

    first = module.package_bundle(config)
    first_sha = first["archive_sha256"]
    second = module.package_bundle(config)

    assert second["archive_sha256"] == first_sha
    assert second["audited_tree_manifest_sha256"] == second[
        "archive_tree_manifest_sha256"
    ]
    assert second["weights_included"] is False


def test_rejects_tree_changed_after_audit(tmp_path):
    module = load_script()
    config = config_for(tmp_path, module)
    (Path(config["bundle_root"]) / "README.md").write_text("changed\n")

    with pytest.raises(module.ReleaseArchiveError, match="changed after audit"):
        module.package_bundle(config)


def test_rejects_failed_audit(tmp_path):
    module = load_script()
    config = config_for(tmp_path, module)
    audit = Path(config["audit_report"])
    payload = json.loads(audit.read_text())
    payload["automatic_gate_passed"] = False
    audit.write_text(json.dumps(payload))

    with pytest.raises(module.ReleaseArchiveError, match="did not pass"):
        module.package_bundle(config)
