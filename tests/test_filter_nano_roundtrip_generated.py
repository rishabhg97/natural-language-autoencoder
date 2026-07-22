import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "filter_nano_roundtrip_generated.py"
    spec = importlib.util.spec_from_file_location("filter_nano_roundtrip_generated", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def record(doc_id: str, n_raw_tokens: int, *, value: str) -> dict:
    return {
        "split": "validation",
        "doc_id": doc_id,
        "n_raw_tokens": n_raw_tokens,
        "controls": {"real": {"generated": value}},
    }


def test_selects_source_records_in_reference_order():
    module = load_script()
    source = [record("doc-1", 2, value="one"), record("doc-2", 3, value="two")]
    reference = [record("doc-2", 3, value="ignored"), record("doc-1", 2, value="ignored")]

    selected = module.select_records(source, reference)

    assert [item["controls"]["real"]["generated"] for item in selected] == ["two", "one"]


def test_rejects_missing_reference_identity():
    module = load_script()

    with pytest.raises(module.GeneratedFilterError, match="missing 1 reference rows"):
        module.select_records(
            [record("doc-1", 2, value="one")],
            [record("doc-2", 3, value="two")],
        )


def test_atomic_jsonl_round_trip(tmp_path):
    module = load_script()
    destination = tmp_path / "selected.jsonl"
    records = [record("doc-1", 2, value="one")]

    module.write_jsonl_atomic(destination, records)

    assert [json.loads(line) for line in destination.read_text().splitlines()] == records
