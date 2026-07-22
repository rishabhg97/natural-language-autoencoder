from __future__ import annotations

from pathlib import Path

import pytest

from observatory.fetch_base import (
    configure_hf_transport,
    download_snapshot_with_retries,
)


def test_configure_hf_transport_defaults_to_proxy_compatible_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HF_HUB_DISABLE_XET", "HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    result = configure_hf_transport()
    assert result == {
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "HF_HUB_ETAG_TIMEOUT": "60",
    }


def test_configure_hf_transport_preserves_explicit_operator_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "120")
    result = configure_hf_transport()
    assert result["HF_HUB_DISABLE_XET"] == "0"
    assert result["HF_HUB_DOWNLOAD_TIMEOUT"] == "1200"
    assert result["HF_HUB_ETAG_TIMEOUT"] == "120"


def test_download_snapshot_retries_with_decreasing_concurrency(tmp_path: Path) -> None:
    workers: list[int] = []

    def fake_download(**kwargs: object) -> str:
        workers.append(int(kwargs["max_workers"]))
        if len(workers) < 3:
            raise RuntimeError("transient proxy failure")
        return str(tmp_path)

    sleeps: list[float] = []
    result = download_snapshot_with_retries(
        fake_download,
        repo_id="public/model",
        revision="abc",
        cache_dir=str(tmp_path / "cache"),
        max_workers=8,
        attempts=4,
        sleep=sleeps.append,
    )
    assert result == tmp_path
    assert workers == [8, 4, 2]
    assert sleeps == [5.0, 10.0]


def test_download_snapshot_raises_after_bounded_attempts(tmp_path: Path) -> None:
    def fail(**_: object) -> str:
        raise RuntimeError("persistent proxy failure")

    with pytest.raises(OSError, match="after 2 attempts"):
        download_snapshot_with_retries(
            fail,
            repo_id="public/model",
            revision="abc",
            cache_dir=str(tmp_path),
            max_workers=2,
            attempts=2,
            sleep=lambda _: None,
        )
