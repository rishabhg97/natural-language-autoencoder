import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "build_nano_content_family_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "build_nano_content_family_manifest", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_build_does_not_require_exposure_coverage(tmp_path):
    module = load_module()
    source = tmp_path / "base.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": "doc-a",
                    "detokenized_text_truncated": "alpha beta gamma delta epsilon",
                },
                {
                    "doc_id": "doc-b",
                    "detokenized_text_truncated": "unrelated source document text",
                },
            ]
        ),
        source,
    )
    output = tmp_path / "manifest.json"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "nano_content_family_build.v1",
                "family_sources": [
                    {
                        "path": str(source),
                        "text_field": "detokenized_text_truncated",
                    }
                ],
                "outputs": {"manifest_json": str(output)},
            }
        )
    )

    result = module.run_build(config)

    assert output.is_file()
    assert result["coverage"] is None
    assert result["coverage_path"] is None
    assert result["manifest"]["stats"]["document_count"] == 2
    assert (
        result["manifest"]["algorithm"]["exact_threshold_closure"]
        == "deterministic_prefix_filter"
    )


def test_manifest_cli_without_coverage_emits_summary(tmp_path, capsys):
    module = load_module()
    source = tmp_path / "base.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": "doc-a",
                    "detokenized_text_truncated": "alpha beta gamma delta epsilon",
                }
            ]
        ),
        source,
    )
    output = tmp_path / "manifest.json"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "nano_content_family_build.v1",
                "family_sources": [
                    {
                        "path": str(source),
                        "text_field": "detokenized_text_truncated",
                    }
                ],
                "outputs": {"manifest_json": str(output)},
            }
        )
    )

    assert module.main([str(config)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["manifest_path"] == str(output)
    assert summary["coverage_path"] is None
    assert "holdout_decision" not in summary


def test_coverage_sources_must_be_configured_as_a_pair(tmp_path):
    module = load_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "nano_content_family_build.v1",
                "family_sources": [{"path": "/data/base.parquet"}],
                "exposure_sources": [{"path": "/data/train.parquet"}],
                "outputs": {"manifest_json": "/out/manifest.json"},
            }
        )
    )

    with pytest.raises(
        module.FunctionalEvaluationError,
        match="must be provided together",
    ):
        module.load_config(config)
