#!/usr/bin/env python3
"""Deterministic source fingerprints for Nano experiment launches."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SOURCE_ROOTS = (
    "scripts",
    "external/natural_language_autoencoders/nla",
    "external/natural_language_autoencoders/configs",
)
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".lock", ".tmp"}
TOKENIZER_FILENAMES = {
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


class SourceProvenanceError(ValueError):
    """Raised when a launch does not match its declared source policy."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(root: Path, files: Iterable[Path]) -> str:
    records = [
        (
            source.relative_to(root).as_posix(),
            source.stat().st_size,
            source.stat().st_mtime_ns,
        )
        for source in files
    ]
    canonical = json.dumps(records, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _tokenizer_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SourceProvenanceError(f"tokenizer root does not exist: {root}")
    files = sorted(
        source
        for source in root.iterdir()
        if source.is_file()
        and (source.name in TOKENIZER_FILENAMES or source.name.startswith("tokenizer."))
    )
    if not files:
        raise SourceProvenanceError(f"no tokenizer files found under {root}")
    return files


def tokenizer_stat_signature(path: str | Path) -> str:
    root = Path(path).resolve()
    return _stat_signature(root, _tokenizer_files(root))


def fingerprint_tokenizer_files(path: str | Path) -> dict[str, Any]:
    """Fingerprint the files that define tokenizer behavior.

    This preserves the byte-stream algorithm historically used by
    ``nano_roundtrip_queue`` so existing publication fingerprints remain
    stable after moving the implementation into the shared provenance module.
    """

    root = Path(path).resolve()
    files = _tokenizer_files(root)
    digest = hashlib.sha256()
    total_bytes = 0
    for source in files:
        relative = source.relative_to(root).as_posix().encode()
        payload = source.read_bytes()
        total_bytes += len(payload)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "root": str(root),
        "files": [source.name for source in files],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "stat_signature": _stat_signature(root, files),
        "sha256": digest.hexdigest(),
    }


def _hf_model_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SourceProvenanceError(f"HF model root does not exist: {root}")

    index_files = sorted(
        source
        for source in root.iterdir()
        if source.is_file()
        and source.name in {
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }
    )
    referenced: set[str] = set()
    for index_path in index_files:
        try:
            payload = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceProvenanceError(
                f"invalid HF model index: {index_path}"
            ) from exc
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SourceProvenanceError(
                f"HF model index has no weight_map: {index_path}"
            )
        referenced.update(str(value) for value in weight_map.values())

    if referenced:
        weight_files = [root / name for name in sorted(referenced)]
        missing = [path for path in weight_files if not path.is_file()]
        if missing:
            raise SourceProvenanceError(
                f"HF model index references missing shards: {missing[:5]}"
            )
    else:
        weight_files = sorted(
            source
            for source in root.iterdir()
            if source.is_file()
            and (
                source.suffix == ".safetensors"
                or source.name == "pytorch_model.bin"
                or (
                    source.name.startswith("pytorch_model-")
                    and source.suffix == ".bin"
                )
            )
        )
    if not weight_files:
        raise SourceProvenanceError(f"no HF model weight files found under {root}")
    empty = [path for path in weight_files if path.stat().st_size <= 0]
    if empty:
        raise SourceProvenanceError(f"HF model contains empty shards: {empty[:5]}")
    return sorted({*index_files, *weight_files}, key=lambda path: path.name)


def hf_model_stat_signature(path: str | Path) -> str:
    root = Path(path).resolve()
    return _stat_signature(root, _hf_model_files(root))


def fingerprint_hf_model_files(
    path: str | Path,
    *,
    workers: int = 1,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Build a deterministic content address for an HF model artifact.

    Files are hashed independently, optionally in parallel, then combined in
    canonical relative-path order. The pre/post stat check rejects concurrent
    or partial writes rather than publishing an ambiguous fingerprint.
    """

    if workers < 1:
        raise SourceProvenanceError("HF fingerprint workers must be at least 1")
    root = Path(path).resolve()
    files = _hf_model_files(root)
    before = {
        source: (source.stat().st_size, source.stat().st_mtime_ns)
        for source in files
    }

    def hash_one(source: Path) -> tuple[Path, str]:
        return source, sha256_file(source, chunk_size=chunk_size)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashed = dict(executor.map(hash_one, files))

    changed = [
        source
        for source, expected in before.items()
        if (source.stat().st_size, source.stat().st_mtime_ns) != expected
    ]
    if changed:
        raise SourceProvenanceError(
            f"HF model changed while fingerprinting: {changed[:5]}"
        )

    digest = hashlib.sha256()
    records = []
    total_bytes = 0
    for source in files:
        relative = source.relative_to(root).as_posix()
        size = before[source][0]
        file_sha256 = hashed[source]
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha256))
        total_bytes += size
        records.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": file_sha256,
            }
        )
    return {
        "root": str(root),
        "algorithm": "sha256(path_length,path,size,file_sha256)",
        "files": records,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "stat_signature": _stat_signature(root, files),
        "sha256": digest.hexdigest(),
    }


def _directory_files(root: Path, *, follow_symlinks: bool = False) -> list[Path]:
    if not root.is_dir():
        raise SourceProvenanceError(f"source root does not exist: {root}")
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or (path.is_symlink() and not follow_symlinks):
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def directory_stat_signature(
    path: str | Path, *, follow_symlinks: bool = False
) -> dict[str, Any]:
    root = Path(path).resolve()
    files = _directory_files(root, follow_symlinks=follow_symlinks)
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(source.stat().st_size for source in files),
        "sha256": _stat_signature(root, files),
    }


def _update_digest_from_file(
    digest: Any,
    source: Path,
    *,
    relative: bytes,
    chunk_size: int = 8 * 1024 * 1024,
) -> int:
    size = source.stat().st_size
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(size.to_bytes(8, "big"))
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return size


def fingerprint_directory(
    path: str | Path, *, label: str, follow_symlinks: bool = False
) -> dict[str, Any]:
    root = Path(path).resolve()
    files = _directory_files(root, follow_symlinks=follow_symlinks)
    digest = hashlib.sha256()
    total_bytes = 0
    for source in files:
        relative = source.relative_to(root).as_posix().encode()
        total_bytes += _update_digest_from_file(
            digest,
            source,
            relative=relative,
        )
    return {
        "label": str(label),
        "root": str(root),
        "followed_symlinks": bool(follow_symlinks),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def fingerprint_and_copy_directory(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    label: str,
    chunk_size: int = 8 * 1024 * 1024,
    workers: int = 1,
    task_size: int = 512 * 1024 * 1024,
    follow_symlinks: bool = False,
) -> dict[str, Any]:
    """Fingerprint a directory while copying each byte exactly once.

    This is intended for staging large immutable checkpoint trees from shared
    storage to local tmpfs. The returned fingerprint is byte-for-byte
    compatible with :func:`fingerprint_directory` for the source tree.
    """

    source_root = Path(source_path).resolve()
    destination_root = Path(destination_path)
    if workers < 1:
        raise SourceProvenanceError("staging workers must be at least 1")
    if task_size < 1:
        raise SourceProvenanceError("staging task size must be at least 1")
    if destination_root.exists():
        raise SourceProvenanceError(
            f"staging destination already exists: {destination_root}"
        )
    temporary_root = destination_root.with_name(destination_root.name + ".tmp")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)

    files = _directory_files(source_root, follow_symlinks=follow_symlinks)
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        if workers == 1:
            for source in files:
                relative_path = source.relative_to(source_root)
                relative = relative_path.as_posix().encode()
                size = source.stat().st_size
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(size.to_bytes(8, "big"))
                destination = temporary_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with source.open("rb") as source_handle, destination.open(
                    "wb"
                ) as destination_handle:
                    while chunk := source_handle.read(chunk_size):
                        digest.update(chunk)
                        destination_handle.write(chunk)
                        copied += len(chunk)
                if copied != size:
                    raise SourceProvenanceError(
                        f"staged byte count mismatch for {source}: {copied} != {size}"
                    )
                total_bytes += copied
        else:
            source_stats: dict[Path, tuple[int, int]] = {}
            tasks: list[tuple[Path, Path, int, int]] = []
            for source in files:
                source_stat = source.stat()
                source_stats[source] = (
                    source_stat.st_size,
                    source_stat.st_mtime_ns,
                )
                destination = temporary_root / source.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as destination_handle:
                    destination_handle.truncate(source_stat.st_size)
                for offset in range(0, source_stat.st_size, task_size):
                    tasks.append(
                        (
                            source,
                            destination,
                            offset,
                            min(task_size, source_stat.st_size - offset),
                        )
                    )

            def copy_range(task: tuple[Path, Path, int, int]) -> int:
                source, destination, offset, length = task
                source_digest = hashlib.sha256()
                copied = 0
                with source.open("rb") as source_handle, destination.open(
                    "r+b"
                ) as destination_handle:
                    source_handle.seek(offset)
                    destination_handle.seek(offset)
                    remaining = length
                    while remaining:
                        chunk = source_handle.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        source_digest.update(chunk)
                        destination_handle.write(chunk)
                        copied += len(chunk)
                        remaining -= len(chunk)
                if copied != length:
                    raise SourceProvenanceError(
                        f"staged range byte count mismatch for {source}: "
                        f"offset={offset} copied={copied} expected={length}"
                    )
                staged_digest = hashlib.sha256()
                with destination.open("rb") as destination_handle:
                    destination_handle.seek(offset)
                    remaining = length
                    while remaining:
                        chunk = destination_handle.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        staged_digest.update(chunk)
                        remaining -= len(chunk)
                if staged_digest.digest() != source_digest.digest():
                    raise SourceProvenanceError(
                        f"staged SHA-256 mismatch for {source} at offset {offset}"
                    )
                return copied

            with ThreadPoolExecutor(max_workers=workers) as executor:
                total_bytes = sum(executor.map(copy_range, tasks))
            for source, (expected_size, expected_mtime_ns) in source_stats.items():
                final_stat = source.stat()
                if (
                    final_stat.st_size != expected_size
                    or final_stat.st_mtime_ns != expected_mtime_ns
                ):
                    raise SourceProvenanceError(
                        f"source changed while staging: {source}"
                    )
            staged_fingerprint = fingerprint_directory(
                temporary_root,
                label=label,
            )
            if staged_fingerprint["total_bytes"] != total_bytes:
                raise SourceProvenanceError(
                    "staged directory byte count does not match copied byte count"
                )
            digest_hex = staged_fingerprint["sha256"]
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root.replace(destination_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return {
        "label": str(label),
        "root": str(source_root),
        "staged_root": str(destination_root.resolve()),
        "followed_symlinks": bool(follow_symlinks),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest() if workers == 1 else digest_hex,
    }


def python_environment_fingerprint(
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    executable = str(python_executable)
    script = (
        "import importlib.metadata,json,sys;"
        "packages=sorted((d.metadata.get('Name') or d.name,d.version) "
        "for d in importlib.metadata.distributions());"
        "print(json.dumps({'executable':sys.executable,'version':sys.version,"
        "'packages':packages},sort_keys=True))"
    )
    completed = subprocess.run(
        [executable, "-c", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    environment = json.loads(completed.stdout)
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return {
        "executable": environment["executable"],
        "python_version": environment["version"],
        "package_count": len(environment["packages"]),
        "packages": environment["packages"],
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _source_files(code_root: Path, roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for relative_root in roots:
        source_root = code_root / relative_root
        if not source_root.is_dir():
            raise SourceProvenanceError(f"source root does not exist: {source_root}")
        for path in source_root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(code_root)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            if path.suffix in IGNORED_SUFFIXES:
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(code_root).as_posix())


def fingerprint_source_tree(
    code_root: str | Path,
    *,
    roots: Iterable[str] = DEFAULT_SOURCE_ROOTS,
) -> dict[str, Any]:
    root = Path(code_root).resolve()
    digest = hashlib.sha256()
    files = _source_files(root, roots)
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        total_bytes += _update_digest_from_file(
            digest,
            path,
            relative=relative,
        )
    return {
        "schema_version": "nano_source_fingerprint.v1",
        "code_root": str(root),
        "roots": list(roots),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def source_file_manifest(
    code_root: str | Path,
    *,
    roots: Iterable[str] = DEFAULT_SOURCE_ROOTS,
) -> dict[str, Any]:
    root = Path(code_root).resolve()
    files = _source_files(root, roots)
    return {
        "schema_version": "nano_source_file_manifest.v1",
        "fingerprint": fingerprint_source_tree(root, roots=roots),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def git_source_state(code_root: str | Path) -> dict[str, Any]:
    root = Path(code_root).resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    try:
        git_root = Path(run("rev-parse", "--show-toplevel")).resolve()
        head = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current") or None
        relative_root = root.relative_to(git_root)
        status = run("status", "--porcelain", "--", str(relative_root))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return {"available": False}
    dirty_paths = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "root": str(git_root),
        "head": head,
        "branch": branch,
        "dirty": bool(dirty_paths),
        "dirty_entry_count": len(dirty_paths),
    }


def collect_source_provenance(
    code_root: str | Path,
    *,
    queue_path: str | Path | None = None,
    roots: Iterable[str] = DEFAULT_SOURCE_ROOTS,
    miles_root: str | Path | None = None,
    miles_patches_root: str | Path | None = None,
    critical_files: Mapping[str, str | Path] | None = None,
    container_image_digest: str | None = None,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "nano_source_provenance.v1",
        "source": fingerprint_source_tree(code_root, roots=roots),
        "git": git_source_state(code_root),
    }
    if queue_path is not None:
        queue = Path(queue_path).resolve()
        payload["queue"] = {
            "path": str(queue),
            "sha256": sha256_file(queue),
        }
    runtime_components: dict[str, Any] = {
        "project_source": payload["source"],
        "python_environment": python_environment_fingerprint(python_executable),
    }
    if miles_root is not None:
        runtime_components["miles"] = fingerprint_directory(
            miles_root,
            label="miles",
        )
    if miles_patches_root is not None:
        runtime_components["miles_patches"] = fingerprint_directory(
            miles_patches_root,
            label="miles_patches",
        )
    critical: dict[str, Any] = {}
    for name, value in sorted((critical_files or {}).items()):
        path = Path(value).resolve()
        if not path.is_file():
            raise SourceProvenanceError(f"critical file does not exist: {name}={path}")
        critical[str(name)] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if queue_path is not None:
        critical.setdefault("queue", dict(payload["queue"]))
    normalized_container_digest = str(container_image_digest or "").strip() or None
    runtime_material = {
        "components": runtime_components,
        "critical_files": critical,
        "container_image_digest": normalized_container_digest,
    }
    runtime_sha256 = hashlib.sha256(
        json.dumps(runtime_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["runtime"] = {
        **runtime_material,
        "schema_version": "nano_runtime_fingerprint.v1",
        "sha256": runtime_sha256,
        "complete": bool(
            (runtime_components.get("miles") or {}).get("file_count")
            and (runtime_components.get("miles_patches") or {}).get("file_count")
            and critical
            and normalized_container_digest
            and (runtime_components.get("python_environment") or {}).get("sha256")
        ),
    }
    return payload


def verify_source_policy(policy: dict[str, Any], provenance: dict[str, Any]) -> None:
    expected = policy.get("expected_code_sha256")
    actual = (provenance.get("source") or {}).get("sha256")
    if expected and str(expected) != str(actual):
        raise SourceProvenanceError(
            f"source fingerprint mismatch: expected {expected}, got {actual}"
        )

    git = provenance.get("git") or {}
    if policy.get("require_git") and not git.get("available"):
        raise SourceProvenanceError("launch policy requires Git metadata")
    expected_commit = policy.get("expected_git_commit")
    frozen_commit = policy.get("frozen_git_commit")
    if expected_commit and frozen_commit and str(expected_commit) != str(frozen_commit):
        raise SourceProvenanceError(
            "expected_git_commit and frozen_git_commit disagree"
        )
    expected_commit = expected_commit or frozen_commit
    if expected_commit and str(git.get("head")) != str(expected_commit):
        raise SourceProvenanceError(
            f"Git commit mismatch: expected {expected_commit}, got {git.get('head')}"
        )
    if policy.get("require_clean_git") and git.get("dirty"):
        raise SourceProvenanceError(
            f"launch policy requires a clean tree; found {git.get('dirty_entry_count')} entries"
        )
    if policy.get("require_expected_fingerprint") and not expected:
        raise SourceProvenanceError("launch policy requires expected_code_sha256")
    runtime = provenance.get("runtime") or {}
    if policy.get("require_complete_runtime") and not runtime.get("complete"):
        raise SourceProvenanceError(
            "launch policy requires a complete runtime fingerprint"
        )
    expected_runtime = policy.get("expected_runtime_sha256")
    if expected_runtime and str(expected_runtime) != str(runtime.get("sha256")):
        raise SourceProvenanceError(
            "runtime fingerprint mismatch: "
            f"expected {expected_runtime}, got {runtime.get('sha256')}"
        )
    required_critical_files = {
        str(name) for name in policy.get("required_critical_files") or []
    }
    missing_critical = sorted(
        required_critical_files - set((runtime.get("critical_files") or {}))
    )
    if missing_critical:
        raise SourceProvenanceError(
            f"runtime fingerprint is missing critical files: {missing_critical}"
        )


def write_provenance(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
