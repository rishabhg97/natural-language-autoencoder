from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_nano_nla_checkpoint_pair_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pair_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def _config(tmp_path: Path) -> dict:
    av = tmp_path / "av"
    ar = tmp_path / "ar"
    av.mkdir()
    ar.mkdir()
    av_fingerprint = tmp_path / "av_fingerprint.json"
    ar_fingerprint = tmp_path / "ar_fingerprint.json"
    _write_json(av_fingerprint, {"model": {"sha256": "av-sha"}})
    _write_json(ar_fingerprint, {"sha256": "ar-sha"})
    verifier = tmp_path / "verifier.json"
    gate = tmp_path / "gate.json"
    _write_json(verifier, {"passed": True})
    _write_json(gate, {"gate": {"passed": True}})
    source = tmp_path / "source.json"
    _write_json(source, {"sha256": "source-sha"})
    return {
        "schema_version": "nano_nla_checkpoint_pair_release.v1",
        "release_id": "r33-clean-sft-pair",
        "claim_scope": "family-disjoint validation directional recovery",
        "limitations": ["No raw-magnitude reconstruction claim."],
        "checkpoints": {
            "av": {
                "path": str(av),
                "fingerprint_report": str(av_fingerprint),
                "fingerprint_field": "model.sha256",
                "expected_fingerprint": "av-sha",
            },
            "ar": {
                "path": str(ar),
                "fingerprint_report": str(ar_fingerprint),
                "expected_fingerprint": "ar-sha",
            },
        },
        "evidence": {
            "av_eval": {"path": str(verifier)},
            "roundtrip": {"path": str(gate), "pass_field": "gate.passed"},
        },
        "artifacts": {"source": {"path": str(source)}},
        "metadata": {"layer": 33},
    }


def test_build_manifest_binds_checkpoints_and_passing_evidence(tmp_path: Path):
    module = _load_module()
    manifest = module.build_manifest(_config(tmp_path))

    assert manifest["qualified"] is True
    assert manifest["checkpoints"]["av"]["fingerprint"] == "av-sha"
    assert manifest["checkpoints"]["ar"]["fingerprint"] == "ar-sha"
    assert manifest["evidence"]["roundtrip"]["pass_value"] is True
    assert len(manifest["artifacts"]["source"]["sha256"]) == 64


def test_build_manifest_rejects_failed_evidence(tmp_path: Path):
    module = _load_module()
    config = _config(tmp_path)
    path = Path(config["evidence"]["av_eval"]["path"])
    _write_json(path, {"passed": False})

    with pytest.raises(module.PairManifestError, match="did not pass"):
        module.build_manifest(config)


def test_build_manifest_rejects_fingerprint_mismatch(tmp_path: Path):
    module = _load_module()
    config = _config(tmp_path)
    config["checkpoints"]["ar"]["expected_fingerprint"] = "wrong"

    with pytest.raises(module.PairManifestError, match="fingerprint mismatch"):
        module.build_manifest(config)


def test_build_manifest_requires_explicit_limitations(tmp_path: Path):
    module = _load_module()
    config = _config(tmp_path)
    config["limitations"] = []

    with pytest.raises(module.PairManifestError, match="non-empty list"):
        module.build_manifest(config)
