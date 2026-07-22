from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "nano_source_provenance.py"
    spec = importlib.util.spec_from_file_location("nano_source_provenance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_streaming_file_digest_matches_legacy_payload_digest(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "weights.bin"
    payload = b"abcdefgh" * 1024
    source.write_bytes(payload)
    relative = b"weights.bin"
    expected = hashlib.sha256()
    expected.update(len(relative).to_bytes(8, "big"))
    expected.update(relative)
    expected.update(len(payload).to_bytes(8, "big"))
    expected.update(payload)
    observed = hashlib.sha256()

    size = module._update_digest_from_file(
        observed,
        source,
        relative=relative,
        chunk_size=17,
    )

    assert size == len(payload)
    assert observed.hexdigest() == expected.hexdigest()


def test_source_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "scripts"
    source.mkdir()
    tracked = source / "runner.py"
    tracked.write_text("print('one')\n")
    (source / "ignored.pyc").write_bytes(b"ignored")

    first = module.fingerprint_source_tree(tmp_path, roots=("scripts",))
    second = module.fingerprint_source_tree(tmp_path, roots=("scripts",))
    tracked.write_text("print('two')\n")
    changed = module.fingerprint_source_tree(tmp_path, roots=("scripts",))

    assert first["sha256"] == second["sha256"]
    assert first["sha256"] != changed["sha256"]
    assert first["file_count"] == 1


def test_source_policy_fails_closed_on_mismatch(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "scripts"
    source.mkdir()
    (source / "runner.py").write_text("pass\n")
    provenance = module.collect_source_provenance(tmp_path, roots=("scripts",))

    with pytest.raises(module.SourceProvenanceError, match="fingerprint mismatch"):
        module.verify_source_policy(
            {"expected_code_sha256": "0" * 64},
            provenance,
        )


def test_frozen_git_commit_is_enforced_as_expected_commit() -> None:
    module = load_module()
    provenance = {
        "source": {"sha256": "a" * 64},
        "git": {"available": True, "head": "b" * 40, "dirty": False},
        "runtime": {},
    }

    with pytest.raises(module.SourceProvenanceError, match="Git commit mismatch"):
        module.verify_source_policy(
            {"frozen_git_commit": "c" * 40},
            provenance,
        )


def test_conflicting_git_commit_aliases_fail_closed() -> None:
    module = load_module()
    provenance = {
        "source": {"sha256": "a" * 64},
        "git": {"available": True, "head": "b" * 40, "dirty": False},
        "runtime": {},
    }

    with pytest.raises(module.SourceProvenanceError, match="disagree"):
        module.verify_source_policy(
            {
                "expected_git_commit": "b" * 40,
                "frozen_git_commit": "c" * 40,
            },
            provenance,
        )


def test_complete_runtime_fingerprint_binds_miles_patches_and_artifacts(tmp_path: Path) -> None:
    module = load_module()
    code_root = tmp_path / "code"
    scripts = code_root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "runner.py").write_text("pass\n")
    miles_root = tmp_path / "miles"
    miles_root.mkdir()
    miles_file = miles_root / "train.py"
    miles_file.write_text("print('miles-v1')\n")
    patches_root = code_root / "external" / "natural_language_autoencoders" / "nla" / "miles_patches"
    patches_root.mkdir(parents=True)
    (patches_root / "actor_patch.py").write_text("PATCH = 1\n")
    queue = tmp_path / "queue.yaml"
    dataset = tmp_path / "rl.parquet"
    package_lock = tmp_path / "packages.lock"
    queue.write_text("items: []\n")
    dataset.write_bytes(b"dataset-v1")
    package_lock.write_text("numpy==2.0\n")

    first = module.collect_source_provenance(
        code_root,
        queue_path=queue,
        roots=("scripts",),
        miles_root=miles_root,
        miles_patches_root=patches_root,
        critical_files={"rl_dataset": dataset, "package_lock": package_lock},
        container_image_digest="sha256:image",
    )
    miles_file.write_text("print('miles-v2')\n")
    changed = module.collect_source_provenance(
        code_root,
        queue_path=queue,
        roots=("scripts",),
        miles_root=miles_root,
        miles_patches_root=patches_root,
        critical_files={"rl_dataset": dataset, "package_lock": package_lock},
        container_image_digest="sha256:image",
    )

    assert first["runtime"]["complete"] is True
    assert first["runtime"]["sha256"] != changed["runtime"]["sha256"]
    assert first["runtime"]["components"]["miles"]["file_count"] == 1
    assert first["runtime"]["components"]["miles_patches"]["file_count"] == 1
    assert first["runtime"]["critical_files"]["rl_dataset"]["sha256"] == module.sha256_file(dataset)


def test_fingerprint_and_copy_directory_matches_canonical_digest(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    (source / "nested").mkdir(parents=True)
    (source / "a.bin").write_bytes(b"a" * 37)
    (source / "nested" / "b.bin").write_bytes(b"b" * 113)

    expected = module.fingerprint_directory(source, label="checkpoint")
    staged = module.fingerprint_and_copy_directory(
        source,
        destination,
        label="checkpoint",
        chunk_size=17,
        workers=2,
        task_size=23,
    )
    copied = module.fingerprint_directory(destination, label="checkpoint")

    assert staged["sha256"] == expected["sha256"] == copied["sha256"]
    assert staged["file_count"] == expected["file_count"] == 2
    assert staged["total_bytes"] == expected["total_bytes"] == 150
    assert (destination / "a.bin").read_bytes() == b"a" * 37
    assert (destination / "nested" / "b.bin").read_bytes() == b"b" * 113


def test_fingerprint_and_copy_directory_can_materialize_symlink_tree(
    tmp_path: Path,
) -> None:
    module = load_module()
    blobs = tmp_path / "blobs"
    source = tmp_path / "snapshot"
    destination = tmp_path / "staged"
    blobs.mkdir()
    source.mkdir()
    (blobs / "weights").write_bytes(b"weights" * 19)
    (source / "model.safetensors").symlink_to(blobs / "weights")

    staged = module.fingerprint_and_copy_directory(
        source,
        destination,
        label="hf_snapshot",
        workers=2,
        task_size=17,
        follow_symlinks=True,
    )
    expected = module.fingerprint_directory(
        source,
        label="hf_snapshot",
        follow_symlinks=True,
    )

    assert staged["sha256"] == expected["sha256"]
    assert staged["followed_symlinks"] is True
    assert not (destination / "model.safetensors").is_symlink()
    assert (destination / "model.safetensors").read_bytes() == b"weights" * 19


def test_hf_model_fingerprint_is_parallel_deterministic_and_content_bound(
    tmp_path: Path,
) -> None:
    module = load_module()
    model = tmp_path / "hf"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"a" * 101)
    (model / "model-00002-of-00002.safetensors").write_bytes(b"b" * 137)
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}\n'
    )

    serial = module.fingerprint_hf_model_files(model, workers=1, chunk_size=17)
    parallel = module.fingerprint_hf_model_files(model, workers=2, chunk_size=19)
    original_signature = module.hf_model_stat_signature(model)
    (model / "model-00002-of-00002.safetensors").write_bytes(b"c" * 137)
    changed = module.fingerprint_hf_model_files(model, workers=2, chunk_size=23)

    assert serial["sha256"] == parallel["sha256"]
    assert serial["sha256"] != changed["sha256"]
    assert serial["file_count"] == 3
    assert original_signature != module.hf_model_stat_signature(model)


def test_hf_model_fingerprint_rejects_missing_index_shard(tmp_path: Path) -> None:
    module = load_module()
    model = tmp_path / "hf"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"missing.safetensors"}}\n'
    )

    with pytest.raises(module.SourceProvenanceError, match="missing shards"):
        module.fingerprint_hf_model_files(model)


def test_complete_runtime_policy_rejects_missing_components(tmp_path: Path) -> None:
    module = load_module()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "runner.py").write_text("pass\n")
    provenance = module.collect_source_provenance(tmp_path, roots=("scripts",))

    with pytest.raises(module.SourceProvenanceError, match="complete runtime"):
        module.verify_source_policy(
            {"require_complete_runtime": True},
            provenance,
        )
