from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_s3.py"
    spec = importlib.util.spec_from_file_location("nano_s3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_aws_prefix(cmd, service: str, operation: str) -> None:
    assert Path(cmd[0]).name == "aws"
    assert cmd[1:5] == ["--endpoint-url", "https://pdx.s8k.io", service, operation]


def test_remote_uri_expands_relative_paths_under_project_prefix():
    nano_s3 = load_script()

    assert (
        nano_s3.remote_uri("checkpoints/r27-av-ar-best/")
        == "s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/checkpoints/r27-av-ar-best/"
    )
    assert nano_s3.remote_uri(None) == "s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/"
    assert nano_s3.remote_uri("s3://other-bucket/path") == "s3://other-bucket/path"


def test_configured_env_sets_endpoint_and_credential_paths_without_secret_values():
    nano_s3 = load_script()

    env = nano_s3.configured_env({"PATH": "/bin", "NO_PROXY": "localhost"})

    assert env["AWS_SHARED_CREDENTIALS_FILE"] == "/workspace/interp/secrets/aws/credentials"
    assert env["AWS_CONFIG_FILE"] == "/workspace/interp/secrets/aws/config"
    assert env["AWS_DEFAULT_REGION"] == "us-east-1"
    assert env["S3_ENDPOINT_URL"] == "https://pdx.s8k.io"
    assert env["AWS_ENDPOINT_URL_S3"] == "https://pdx.s8k.io"
    assert env["NO_PROXY"] == "localhost"
    assert env["PATH"] == "/bin"


def test_configured_env_respects_local_credential_and_binary_overrides(monkeypatch):
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/Users/me/.aws/nano/credentials")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/Users/me/.aws/nano/config")
    monkeypatch.setenv("NANO_S3_AWS_BIN", "/opt/homebrew/bin/aws")
    nano_s3 = load_script()

    env = nano_s3.configured_env({"PATH": "/bin"})

    assert env["AWS_SHARED_CREDENTIALS_FILE"] == "/Users/me/.aws/nano/credentials"
    assert env["AWS_CONFIG_FILE"] == "/Users/me/.aws/nano/config"
    assert nano_s3.aws_base()[0] == "/opt/homebrew/bin/aws"


def test_build_sync_up_command_uses_project_endpoint_and_dryrun():
    nano_s3 = load_script()

    cmd = nano_s3.build_sync_command(
        "up",
        "/workspace/interp/outputs/best",
        "checkpoints/r27-av-ar-best/ar/",
        dry_run=True,
        delete=False,
        extra_args=["--exclude", "*.tmp"],
    )

    assert_aws_prefix(cmd, "s3", "sync")
    assert cmd[5:7] == [
        "/workspace/interp/outputs/best",
        "s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/checkpoints/r27-av-ar-best/ar/",
    ]
    assert "--dryrun" in cmd
    assert "--no-progress" in cmd
    assert cmd[-2:] == ["--exclude", "*.tmp"]


def test_build_exists_prefix_command_uses_s3api_bucket_and_prefix():
    nano_s3 = load_script()

    cmd = nano_s3.build_exists_prefix_command("checkpoints/r27-av-ar-best/", max_items=2)

    assert_aws_prefix(cmd, "s3api", "list-objects-v2")
    assert "--bucket" in cmd
    assert cmd[cmd.index("--bucket") + 1] == "team-ipp-trustworthy-ai"
    assert "--prefix" in cmd
    assert cmd[cmd.index("--prefix") + 1] == "nano30b-nla-pilot/checkpoints/r27-av-ar-best/"
    assert "--max-items" in cmd
    assert cmd[cmd.index("--max-items") + 1] == "2"
    assert "--query" in cmd
    assert cmd[cmd.index("--query") + 1] == "KeyCount"


def test_parse_prefix_key_count_distinguishes_empty_and_present():
    nano_s3 = load_script()

    assert nano_s3.parse_key_count("0\n") == 0
    assert nano_s3.parse_key_count("3\n") == 3
    assert nano_s3.parse_key_count("None\n") == 0


def test_transfer_config_text_sets_parallel_s3_upload_options():
    nano_s3 = load_script()

    text = nano_s3.transfer_config_text(
        max_concurrent_requests=48,
        multipart_chunksize_mb=96,
        multipart_threshold_mb=64,
    )

    assert "[default]" in text
    assert "region = us-east-1" in text
    assert "max_concurrent_requests = 48" in text
    assert "multipart_chunksize = 96MB" in text
    assert "multipart_threshold = 64MB" in text


def test_wrap_with_timeout_uses_gnu_timeout_with_kill_after(monkeypatch):
    nano_s3 = load_script()
    monkeypatch.setattr(nano_s3.shutil, "which", lambda name: "/usr/bin/timeout" if name == "timeout" else None)

    assert nano_s3.wrap_with_timeout(["aws", "s3", "ls"], 30) == [
        "timeout",
        "--kill-after",
        "5s",
        "30s",
        "aws",
        "s3",
        "ls",
    ]
    assert nano_s3.wrap_with_timeout(["aws"], None) == ["aws"]
    assert nano_s3.wrap_with_timeout(["aws"], 0) == ["aws"]


def test_wrap_with_timeout_degrades_when_gnu_timeout_is_unavailable(monkeypatch):
    nano_s3 = load_script()
    monkeypatch.setattr(nano_s3.shutil, "which", lambda name: None)

    assert nano_s3.wrap_with_timeout(["aws", "s3", "ls"], 30) == ["aws", "s3", "ls"]
