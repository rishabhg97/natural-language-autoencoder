import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_nano_public_release_bundle.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "build_nano_public_release_bundle", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def base_config(source: Path, output: Path) -> dict:
    return {
        "schema_version": "nano_public_release_bundle.v1",
        "source_root": str(source),
        "output_root": str(output),
        "release_id": "test-release",
        "claim_boundary": "test only",
        "include_globs": ["docs/**", "scripts/*.py"],
        "exclude_globs": ["docs/private/**"],
        "replace_existing": True,
    }


def test_builds_redacted_manifest_bound_tree(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    output = tmp_path / "bundle"
    (source / "docs").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "docs" / "result.md").write_text(
        "owner /Users/alice/project and /workspace/interp/data plus "
        "s3://team-ipp-example/nano30b-nla-pilot/evidence/x\n"
    )
    (source / "scripts" / "run.py").write_text("print('ok')\n")

    report = module.build_bundle(base_config(source, output))

    staged = (output / "docs" / "result.md").read_text()
    assert "/Users/alice/" not in staged
    assert "s3://team-ipp-example" not in staged
    assert "${NANO_LOCAL_HOME}/" in staged
    assert "${NANO_INTERP_ROOT}/" in staged
    assert "${NANO_INTERNAL_S3_ROOT}/evidence/x" in staged
    assert report["weights_included"] is False
    manifest = json.loads((output / module.MANIFEST_NAME).read_text())
    assert manifest["file_count"] == 2
    assert manifest["tree_manifest_sha256"] == report["tree_manifest_sha256"]


def test_copies_extra_evidence_to_safe_destination(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    output = tmp_path / "bundle"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "readme.md").write_text("ok\n")
    evidence = tmp_path / "report.json"
    evidence.write_text('{"path":"/workspace/models/checkpoint"}\n')
    config = base_config(source, output)
    config["extra_files"] = [
        {"source": str(evidence), "destination": "evidence/report.json"}
    ]

    module.build_bundle(config)

    assert "${NANO_MODEL_ROOT}/checkpoint" in (
        output / "evidence" / "report.json"
    ).read_text()


def test_rejects_binary_and_unsafe_destinations(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    output = tmp_path / "bundle"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "bad.md").write_bytes(b"bad\x00data")

    with pytest.raises(module.ReleaseBundleError, match="binary"):
        module.build_bundle(base_config(source, output))

    (source / "docs" / "bad.md").write_text("ok\n")
    config = base_config(source, output)
    config["extra_files"] = [
        {"source": str(source / "docs" / "bad.md"), "destination": "../x"}
    ]
    with pytest.raises(module.ReleaseBundleError, match="unsafe"):
        module.build_bundle(config)
