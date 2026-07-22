from __future__ import annotations

from pathlib import Path

import pytest

from observatory.common import ObservatoryConfigError
from observatory.prepare_runtime import (
    _ExclusiveLock,
    _copy_tree_parallel,
    choose_token_ids,
    stage_hf_checkpoint,
    write_source_parquet,
)


class _Tokenizer:
    bos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [len(word) for word in text.split()]}


def test_choose_token_ids_prefers_exact_candidate() -> None:
    ids, source, lengths = choose_token_ids(
        _Tokenizer(),
        [("short", "a bb"), ("exact", "a bb ccc")],
        expected_length=3,
    )
    assert ids == [1, 2, 3]
    assert source == "exact"
    assert lengths == {"short": 2, "exact": 3}


def test_choose_token_ids_can_restore_bos() -> None:
    ids, source, _ = choose_token_ids(
        _Tokenizer(), [("source", "a bb")], expected_length=3
    )
    assert ids == [99, 1, 2]
    assert source == "source+bos"


def test_choose_token_ids_fails_when_no_candidate_matches() -> None:
    with pytest.raises(ObservatoryConfigError, match="could not reproduce"):
        choose_token_ids(_Tokenizer(), [("source", "a")], expected_length=5)


def test_copy_tree_parallel_preserves_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "first.bin").write_bytes(b"first")
    (source / "nested" / "second.bin").write_bytes(b"second")
    destination = tmp_path / "destination"
    report = _copy_tree_parallel(source, destination, workers=2)
    assert report == {"files": 2, "bytes": 11, "workers": 2}
    assert (destination / "first.bin").read_bytes() == b"first"
    assert (destination / "nested" / "second.bin").read_bytes() == b"second"


def test_exclusive_lock_rejects_second_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "prepare.lock"
    with _ExclusiveLock(lock_path):
        with pytest.raises(ObservatoryConfigError, match="another Observatory"):
            with _ExclusiveLock(lock_path):
                pass
    assert not lock_path.exists()


def test_write_source_parquet_preserves_native_chat_messages(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "source.parquet"
    write_source_parquet(
        path,
        [
            {
                "row_index": 1,
                "doc_id": "doc-1",
                "n_raw_tokens": 2,
                "token_position": 1,
                "token_id": 7,
                "token_text": "token",
                "detokenized_text_truncated": "source",
                "token_ids_prefix": [6, 7],
                "activation_vector": [0.0] * 2688,
                "activation_layer": 33,
                "split": "validation",
                "content_family_id": "family-1",
                "api_explanation": "explanation",
                "prompt": [{"role": "user", "content": "<INJECT>"}],
                "response": "<explanation>answer</explanation>",
            }
        ],
    )
    table = pq.read_table(path, columns=["prompt", "response"])
    row = table.to_pylist()[0]
    assert row["prompt"] == [{"role": "user", "content": "<INJECT>"}]
    assert row["response"] == "<explanation>answer</explanation>"


def test_stage_hf_checkpoint_copies_and_reuses_valid_weights(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint" / "hf"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    destination = tmp_path / "staged"

    copied = stage_hf_checkpoint(
        source_checkpoint=source.parent,
        output_hf=destination,
        force=False,
        copy_workers=2,
    )
    assert copied["status"] == "copied"
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    reused = stage_hf_checkpoint(
        source_checkpoint=source.parent,
        output_hf=destination,
        force=False,
        copy_workers=2,
    )
    assert reused["status"] == "reused"
