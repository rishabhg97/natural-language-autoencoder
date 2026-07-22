import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_nano_release_text as audit  # noqa: E402


def _generated(path: pathlib.Path, texts: list[str]) -> None:
    rows = []
    for index, text in enumerate(texts):
        rows.append(
            {
                "split": "validation",
                "row_index": index,
                "doc_id": f"doc-{index}",
                "content_family_id": f"family-{index}",
                "controls": {"real": {"parsed": {"explanation": text}}},
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _panel(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"splits": {"validation": {"rows": rows}}}) + "\n"
    )


def _config(generated: pathlib.Path, panel: pathlib.Path) -> dict:
    return {
        "schema_version": audit.SCHEMA_VERSION,
        "paths": {
            "generated_jsonl": [{"name": "validation", "path": str(generated)}],
            "panel_json": str(panel),
            "output_json": str(panel.parent / "report.json"),
        },
        "protocol": {
            "generated_text_path": "controls.real.parsed.explanation",
            "min_copy_block_words": 3,
            "max_contiguous_source_copy_words": 8,
            "max_source_copy_fraction": 0.5,
            "min_words_for_copy_fraction": 8,
            "fail_sensitive_kinds": ["email", "private_key", "aws_access_key"],
            "fail_on_source_copy": True,
            "fail_on_reference_copy": False,
        },
    }


def test_clean_text_passes_and_keeps_manual_gate(tmp_path):
    generated = tmp_path / "generated.jsonl"
    panel = tmp_path / "panel.json"
    _generated(generated, ["A short summary of the source material."])
    _panel(
        panel,
        [
            {
                "split": "validation",
                "row_index": 0,
                "doc_id": "doc-0",
                "candidate_text": "A short summary of the source material.",
                "source_text": "The underlying document discusses an unrelated topic.",
                "reference_text": "A concise teacher description.",
            }
        ],
    )
    report = audit.run_audit(_config(generated, panel))
    assert report["automatic_gate_passed"] is True
    assert report["human_review_required"] is True
    assert report["generated_text"]["row_count"] == 1
    assert report["qualitative_panel"]["copy_flagged_rows"] == 0


def test_sensitive_values_are_flagged_but_never_copied_to_report(tmp_path):
    generated = tmp_path / "generated.jsonl"
    panel = tmp_path / "panel.json"
    # Keep the release-auditor regression without tripping repository secret scanners.
    secret = "AKIA" + "ABCDEFGHIJKLMNOP"
    email = "person@example.com"
    _generated(generated, [f"Contact {email}; credential {secret}."])
    _panel(panel, [])
    report = audit.run_audit(_config(generated, panel))
    rendered = json.dumps(report)
    assert report["automatic_gate_passed"] is False
    assert report["generated_text"]["finding_counts"]["email"] == 1
    assert report["generated_text"]["finding_counts"]["aws_access_key"] == 1
    assert email not in rendered
    assert secret not in rendered


def test_luhn_and_private_network_detection():
    counts = audit.scan_sensitive_text(
        "Card 4111 1111 1111 1111 reached 10.0.0.7; 1234 5678 9012 3456 did not."
    )
    assert counts["payment_card"] == 1
    assert counts["private_ipv4"] == 1


def test_source_copy_threshold_fails(tmp_path):
    generated = tmp_path / "generated.jsonl"
    panel = tmp_path / "panel.json"
    copied = "one two three four five six seven eight nine ten"
    _generated(generated, [copied])
    _panel(
        panel,
        [
            {
                "split": "validation",
                "row_index": 0,
                "doc_id": "doc-0",
                "candidate_text": copied,
                "source_text": f"prefix {copied} suffix",
                "reference_text": "different teacher wording",
            }
        ],
    )
    report = audit.run_audit(_config(generated, panel))
    assert report["automatic_gate_passed"] is False
    assert report["qualitative_panel"]["failing_copy_rows"] == 1
    finding = report["qualitative_panel"]["copy_findings"][0]
    assert finding["source"]["longest_contiguous_match_words"] == 10


def test_duplicate_generated_identity_fails_closed(tmp_path):
    generated = tmp_path / "generated.jsonl"
    panel = tmp_path / "panel.json"
    _generated(generated, ["first"])
    _panel(panel, [])
    config = _config(generated, panel)
    config["paths"]["generated_jsonl"].append(
        {"name": "duplicate", "path": str(generated)}
    )
    with pytest.raises(audit.ReleaseTextAuditError, match="duplicate generated row"):
        audit.run_audit(config)


def test_wrong_schema_fails_closed(tmp_path):
    with pytest.raises(audit.ReleaseTextAuditError, match="schema_version"):
        audit.run_audit({"schema_version": "wrong"})
