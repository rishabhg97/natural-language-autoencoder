from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "harden_nano_roundtrip_report.py"
    spec = importlib.util.spec_from_file_location("harden_nano_roundtrip_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_harden_report_adds_doc_keys_and_dataset_hashes(tmp_path: Path) -> None:
    module = load_module()
    report_path = tmp_path / "report.json"
    generated_path = tmp_path / "generated.jsonl"
    report_path.write_text(json.dumps({"splits": {"validation": {"row_indices": [7]}}}))
    generated_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "row_index": 7,
                "doc_id": "doc-1",
                "token_position": 8,
            }
        )
        + "\n"
    )
    datasets = {}
    for name in ("train", "validation", "test"):
        datasets[name] = tmp_path / f"{name}.parquet"
        datasets[name].write_bytes(name.encode())

    report = module.harden_report(
        report_json=report_path,
        generated_jsonl=generated_path,
        train_parquet=datasets["train"],
        validation_parquet=datasets["validation"],
        test_parquet=datasets["test"],
    )

    assert report["splits"]["validation"]["doc_ids"] == ["doc-1"]
    assert report["splits"]["validation"]["row_keys"] == [
        {"doc_id": "doc-1", "token_position": 8}
    ]
    assert len(report["dataset_provenance"]["train"]["sha256"]) == 64
