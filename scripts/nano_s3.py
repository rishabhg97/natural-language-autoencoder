#!/usr/bin/env python3
"""Repeatable S3 helper for Nano30B NLA RunAI checkpoint/artifact syncs.

This script intentionally hardcodes the RunAI PVC credential paths and S3
endpoint used by the project. It prints paths and command status, never secret
contents.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


ENDPOINT_URL = "https://pdx.s8k.io"
BUCKET = "team-ipp-trustworthy-ai"
PROJECT_PREFIX = "nano30b-nla-pilot"
PROJECT_URI = f"s3://{BUCKET}/{PROJECT_PREFIX}/"
CREDENTIALS_FILE = (
    os.environ.get("NANO_S3_AWS_SHARED_CREDENTIALS_FILE")
    or os.environ.get("AWS_SHARED_CREDENTIALS_FILE")
    or "/workspace/interp/secrets/aws/credentials"
)
CONFIG_FILE = (
    os.environ.get("NANO_S3_AWS_CONFIG_FILE")
    or os.environ.get("AWS_CONFIG_FILE")
    or "/workspace/interp/secrets/aws/config"
)
DEFAULT_REGION = "us-east-1"
DEFAULT_CONNECT_TIMEOUT = "10"
DEFAULT_READ_TIMEOUT = "60"
AWS_BIN = os.environ.get("NANO_S3_AWS_BIN") or "/workspace/interp/.venv/bin/aws"
DEFAULT_MAX_CONCURRENT_REQUESTS = 64
DEFAULT_MULTIPART_CHUNKSIZE_MB = 64
DEFAULT_MULTIPART_THRESHOLD_MB = 64


def remote_uri(value: str | None = None) -> str:
    """Return an S3 URI, expanding project-relative prefixes."""

    if value is None or value == "":
        return PROJECT_URI
    if value.startswith("s3://"):
        return value
    prefix = value.lstrip("/")
    return f"{PROJECT_URI}{prefix}"


def split_project_uri(value: str | None = None) -> tuple[str, str]:
    """Return bucket and key prefix for an S3 URI or project-relative path."""

    uri = remote_uri(value)
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3 URI after normalization, got {uri!r}")
    tail = uri[len("s3://") :]
    if "/" not in tail:
        return tail, ""
    bucket, key = tail.split("/", 1)
    return bucket, key


def configured_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an AWS CLI environment without exposing secret values."""

    env = dict(os.environ if base is None else base)
    env["AWS_SHARED_CREDENTIALS_FILE"] = CREDENTIALS_FILE
    env["AWS_CONFIG_FILE"] = CONFIG_FILE
    env["AWS_DEFAULT_REGION"] = DEFAULT_REGION
    env["S3_ENDPOINT_URL"] = ENDPOINT_URL
    env["AWS_ENDPOINT_URL_S3"] = ENDPOINT_URL
    env["AWS_PAGER"] = ""
    env["AWS_EC2_METADATA_DISABLED"] = "true"
    env.setdefault("AWS_CLI_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT)
    env.setdefault("AWS_CLI_READ_TIMEOUT", DEFAULT_READ_TIMEOUT)
    existing_no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    no_proxy_parts = [part for part in existing_no_proxy.split(",") if part]
    if env.get("NANO_S3_DIRECT_NO_PROXY") == "1":
        for part in ("pdx.s8k.io", ".s8k.io"):
            if part not in no_proxy_parts:
                no_proxy_parts.append(part)
    if no_proxy_parts:
        env["NO_PROXY"] = ",".join(no_proxy_parts)
        env["no_proxy"] = env["NO_PROXY"]
    return env


def transfer_config_text(
    *,
    max_concurrent_requests: int,
    multipart_chunksize_mb: int,
    multipart_threshold_mb: int,
) -> str:
    """Return an AWS config file body with high-throughput S3 transfer settings."""

    return "\n".join(
        [
            "[default]",
            f"region = {DEFAULT_REGION}",
            "s3 =",
            f"    max_concurrent_requests = {max_concurrent_requests}",
            f"    multipart_chunksize = {multipart_chunksize_mb}MB",
            f"    multipart_threshold = {multipart_threshold_mb}MB",
            "",
        ]
    )


def write_transfer_config(
    *,
    max_concurrent_requests: int,
    multipart_chunksize_mb: int,
    multipart_threshold_mb: int,
) -> str:
    """Write a temporary AWS config for one transfer command."""

    handle = tempfile.NamedTemporaryFile("w", prefix="nano-s3-aws-config-", suffix=".ini", delete=False)
    with handle:
        handle.write(
            transfer_config_text(
                max_concurrent_requests=max_concurrent_requests,
                multipart_chunksize_mb=multipart_chunksize_mb,
                multipart_threshold_mb=multipart_threshold_mb,
            )
        )
    return handle.name


def aws_base() -> list[str]:
    if os.environ.get("NANO_S3_AWS_BIN"):
        executable = AWS_BIN
    else:
        executable = AWS_BIN if Path(AWS_BIN).is_file() else "aws"
    return [executable, "--endpoint-url", ENDPOINT_URL]


def build_sync_command(
    direction: str,
    source: str,
    dest: str,
    *,
    dry_run: bool = False,
    delete: bool = False,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    if direction == "up":
        src = source
        dst = remote_uri(dest)
    elif direction == "down":
        src = remote_uri(source)
        dst = dest
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    cmd = aws_base() + ["s3", "sync", src, dst, "--no-progress"]
    if dry_run:
        cmd.append("--dryrun")
    if delete:
        cmd.append("--delete")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_cp_command(
    direction: str,
    source: str,
    dest: str,
    *,
    recursive: bool = False,
    dry_run: bool = False,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    if direction == "up":
        src = source
        dst = remote_uri(dest)
    elif direction == "down":
        src = remote_uri(source)
        dst = dest
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    cmd = aws_base() + ["s3", "cp", src, dst, "--no-progress"]
    if recursive:
        cmd.append("--recursive")
    if dry_run:
        cmd.append("--dryrun")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_list_command(remote: str | None, *, recursive: bool = False, summarize: bool = False) -> list[str]:
    cmd = aws_base() + ["s3", "ls", remote_uri(remote)]
    if recursive:
        cmd.append("--recursive")
    if summarize:
        cmd.append("--summarize")
    return cmd


def build_exists_prefix_command(remote: str | None, *, max_items: int = 1) -> list[str]:
    bucket, prefix = split_project_uri(remote)
    return aws_base() + [
        "s3api",
        "list-objects-v2",
        "--bucket",
        bucket,
        "--prefix",
        prefix,
        "--max-items",
        str(max_items),
        "--query",
        "KeyCount",
        "--output",
        "text",
    ]


def build_exists_object_command(remote: str) -> list[str]:
    bucket, key = split_project_uri(remote)
    return aws_base() + ["s3api", "head-object", "--bucket", bucket, "--key", key]


def wrap_with_timeout(cmd: Sequence[str], timeout: int | None) -> list[str]:
    if timeout is None or timeout <= 0:
        return list(cmd)
    if shutil.which("timeout") is None:
        return list(cmd)
    return ["timeout", "--kill-after", "5s", f"{timeout}s", *cmd]


def print_env_status() -> int:
    env = configured_env()
    print(f"endpoint: {ENDPOINT_URL}")
    print(f"project_uri: {PROJECT_URI}")
    print(f"credentials_file: {CREDENTIALS_FILE} exists={Path(CREDENTIALS_FILE).is_file()}")
    print(f"config_file: {CONFIG_FILE} exists={Path(CONFIG_FILE).is_file()}")
    print(f"region: {env['AWS_DEFAULT_REGION']}")
    print("secret_values: not printed")
    return 0


def run_command(
    cmd: Sequence[str],
    *,
    timeout: int | None = None,
    dry_print: bool = False,
    max_concurrent_requests: int | None = None,
    multipart_chunksize_mb: int | None = None,
    multipart_threshold_mb: int | None = None,
) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_print:
        return 0
    config_path: str | None = None
    env = configured_env()
    if max_concurrent_requests is not None:
        config_path = write_transfer_config(
            max_concurrent_requests=max_concurrent_requests,
            multipart_chunksize_mb=multipart_chunksize_mb or DEFAULT_MULTIPART_CHUNKSIZE_MB,
            multipart_threshold_mb=multipart_threshold_mb or DEFAULT_MULTIPART_THRESHOLD_MB,
        )
        env["AWS_CONFIG_FILE"] = config_path
        print(
            "transfer_config: "
            f"max_concurrent_requests={max_concurrent_requests} "
            f"multipart_chunksize={multipart_chunksize_mb or DEFAULT_MULTIPART_CHUNKSIZE_MB}MB "
            f"multipart_threshold={multipart_threshold_mb or DEFAULT_MULTIPART_THRESHOLD_MB}MB",
            flush=True,
        )
    try:
        completed = subprocess.run(wrap_with_timeout(cmd, timeout), env=env, check=False)
    finally:
        if config_path is not None:
            try:
                Path(config_path).unlink()
            except FileNotFoundError:
                pass
    return int(completed.returncode)


def run_capture(cmd: Sequence[str], *, timeout: int | None = None) -> tuple[int, str]:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(
        wrap_with_timeout(cmd, timeout),
        env=configured_env(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return int(completed.returncode), output


def parse_key_count(output: str) -> int:
    text = (output or "").strip()
    if not text or text.lower() == "none":
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def cmd_env(_: argparse.Namespace) -> int:
    return print_env_status()


def cmd_ls(args: argparse.Namespace) -> int:
    return run_command(build_list_command(args.remote, recursive=args.recursive, summarize=args.summarize), timeout=args.timeout)


def cmd_exists(args: argparse.Namespace) -> int:
    if args.object:
        rc = run_command(build_exists_object_command(args.remote), timeout=args.timeout)
        print("exists_object=true" if rc == 0 else "exists_object=false")
        return rc
    rc, output = run_capture(build_exists_prefix_command(args.remote, max_items=args.max_items), timeout=args.timeout)
    count = parse_key_count(output)
    exists = rc == 0 and count > 0
    print(f"exists_prefix={'true' if exists else 'false'} key_count={count} rc={rc}")
    return 0 if exists else 1


def cmd_sync_up(args: argparse.Namespace) -> int:
    return run_command(
        build_sync_command("up", args.local, args.remote, dry_run=args.dry_run, delete=args.delete, extra_args=args.aws_args),
        timeout=None if args.timeout == 0 else args.timeout,
        dry_print=args.print_only,
        max_concurrent_requests=args.max_concurrent_requests,
        multipart_chunksize_mb=args.multipart_chunksize_mb,
        multipart_threshold_mb=args.multipart_threshold_mb,
    )


def cmd_sync_down(args: argparse.Namespace) -> int:
    return run_command(
        build_sync_command("down", args.remote, args.local, dry_run=args.dry_run, delete=args.delete, extra_args=args.aws_args),
        timeout=None if args.timeout == 0 else args.timeout,
        dry_print=args.print_only,
        max_concurrent_requests=args.max_concurrent_requests,
        multipart_chunksize_mb=args.multipart_chunksize_mb,
        multipart_threshold_mb=args.multipart_threshold_mb,
    )


def cmd_cp_up(args: argparse.Namespace) -> int:
    return run_command(
        build_cp_command(
            "up",
            args.local,
            args.remote,
            recursive=args.recursive,
            dry_run=args.dry_run,
            extra_args=args.aws_args,
        ),
        timeout=None if args.timeout == 0 else args.timeout,
        dry_print=args.print_only,
        max_concurrent_requests=args.max_concurrent_requests,
        multipart_chunksize_mb=args.multipart_chunksize_mb,
        multipart_threshold_mb=args.multipart_threshold_mb,
    )


def cmd_cp_down(args: argparse.Namespace) -> int:
    return run_command(
        build_cp_command(
            "down",
            args.remote,
            args.local,
            recursive=args.recursive,
            dry_run=args.dry_run,
            extra_args=args.aws_args,
        ),
        timeout=None if args.timeout == 0 else args.timeout,
        dry_print=args.print_only,
        max_concurrent_requests=args.max_concurrent_requests,
        multipart_chunksize_mb=args.multipart_chunksize_mb,
        multipart_threshold_mb=args.multipart_threshold_mb,
    )


def add_common_transfer_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Pass --dryrun to aws without changing S3/local files.")
    parser.add_argument("--print-only", action="store_true", help="Print the aws command without executing it.")
    parser.add_argument("--timeout", type=int, default=0, help="Command timeout in seconds. 0 disables timeout.")
    parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--multipart-chunksize-mb", type=int, default=DEFAULT_MULTIPART_CHUNKSIZE_MB)
    parser.add_argument("--multipart-threshold-mb", type=int, default=DEFAULT_MULTIPART_THRESHOLD_MB)
    parser.add_argument("--aws-args", nargs=argparse.REMAINDER, default=[], help="Extra args passed to aws after --.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    env_p = sub.add_parser("env", help="Print non-secret S3 configuration and credential file presence.")
    env_p.set_defaults(func=cmd_env)

    ls_p = sub.add_parser("ls", help="List a project-relative or full S3 URI.")
    ls_p.add_argument("remote", nargs="?", default=None)
    ls_p.add_argument("--recursive", action="store_true")
    ls_p.add_argument("--summarize", action="store_true")
    ls_p.add_argument("--timeout", type=int, default=120)
    ls_p.set_defaults(func=cmd_ls)

    exists_p = sub.add_parser("exists", help="Check whether an S3 object or prefix exists.")
    exists_p.add_argument("remote")
    exists_p.add_argument("--object", action="store_true", help="Use head-object instead of prefix listing.")
    exists_p.add_argument("--max-items", type=int, default=1)
    exists_p.add_argument("--timeout", type=int, default=60)
    exists_p.set_defaults(func=cmd_exists)

    up_p = sub.add_parser("sync-up", help="Sync a local directory to project S3.")
    up_p.add_argument("local")
    up_p.add_argument("remote")
    up_p.add_argument("--delete", action="store_true")
    add_common_transfer_flags(up_p)
    up_p.set_defaults(func=cmd_sync_up)

    down_p = sub.add_parser("sync-down", help="Sync a project S3 prefix to a local directory.")
    down_p.add_argument("remote")
    down_p.add_argument("local")
    down_p.add_argument("--delete", action="store_true")
    add_common_transfer_flags(down_p)
    down_p.set_defaults(func=cmd_sync_down)

    cp_up_p = sub.add_parser("cp-up", help="Copy a local file or directory to project S3.")
    cp_up_p.add_argument("local")
    cp_up_p.add_argument("remote")
    cp_up_p.add_argument("--recursive", action="store_true")
    add_common_transfer_flags(cp_up_p)
    cp_up_p.set_defaults(func=cmd_cp_up)

    cp_down_p = sub.add_parser("cp-down", help="Copy an S3 object or prefix to a local path.")
    cp_down_p.add_argument("remote")
    cp_down_p.add_argument("local")
    cp_down_p.add_argument("--recursive", action="store_true")
    add_common_transfer_flags(cp_down_p)
    cp_down_p.set_defaults(func=cmd_cp_down)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
