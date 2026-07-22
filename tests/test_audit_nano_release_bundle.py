from pathlib import Path

from scripts.audit_nano_release_bundle import audit_bundle


def _config(root: Path) -> dict:
    return {
        "schema_version": "nano_release_bundle_audit.v1",
        "root": str(root),
        "fail_pattern_kinds": [
            "aws_access_key",
            "internal_s3",
            "runai_workspace_path",
        ],
    }


def test_bundle_audit_passes_clean_source_tree(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("print('clean')\n")
    report = audit_bundle(_config(tmp_path))

    assert report["automatic_gate_passed"] is True
    assert report["file_count"] == 1
    assert len(report["tree_manifest_sha256"]) == 64


def test_bundle_audit_reports_kinds_without_secret_values(tmp_path: Path) -> None:
    secret = "AKIA" + "A" * 16
    (tmp_path / "config.txt").write_text(
        f"token={secret}\npath=/workspace/interp/data\n"
        "bucket=s3://team-ipp-private/example\n"
    )
    report = audit_bundle(_config(tmp_path))
    encoded = str(report)

    assert report["automatic_gate_passed"] is False
    assert secret not in encoded
    finding = report["failed_finding_files"][0]
    assert finding["path"] == "config.txt"
    assert set(finding["failed_kinds"]) == {
        "aws_access_key",
        "internal_s3",
        "runai_workspace_path",
    }


def test_bundle_audit_rejects_heavyweight_checkpoint_file(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not a real checkpoint")
    report = audit_bundle(_config(tmp_path))

    assert report["automatic_gate_passed"] is False
    assert report["forbidden_paths"][0]["path"] == "model.safetensors"


def test_bundle_audit_allows_bounded_synthetic_fixture(tmp_path: Path) -> None:
    secret = "AKIA" + "A" * 16
    (tmp_path / "fixture.py").write_text(secret + "\n")
    config = _config(tmp_path)
    config["allowed_findings"] = [
        {
            "path_glob": "fixture.py",
            "kinds": ["aws_access_key"],
            "max_count": 1,
        }
    ]
    report = audit_bundle(config)

    assert report["automatic_gate_passed"] is True
    assert report["finding_files"][0]["allowed_kinds"] == ["aws_access_key"]
