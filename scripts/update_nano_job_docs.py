#!/usr/bin/env python3
"""Small updater for Nano AV queue/history Markdown files.

This is intentionally narrow: it supports the common mechanical edits the
RunAI monitor needs without forcing an agent to load and rewrite long files.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
TRACKER = REPO_DIR / "docs" / "nano_av_job_tracker.md"
HISTORY = REPO_DIR / "docs" / "nano_av_run_history.md"
VALID_STATUSES = {
    "draft",
    "ready",
    "running",
    "completed",
    "invalid",
    "failed",
    "held",
    "cancelled",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _stdin_or_file(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def append_history(args: argparse.Namespace) -> int:
    heading = args.heading.strip()
    if not heading.startswith("### "):
        heading = f"### {args.status}: {heading}"
    if not any(heading.startswith(f"### {status}:") for status in VALID_STATUSES):
        raise SystemExit(f"history heading must use a known status: {heading}")

    body = _stdin_or_file(args.entry_file).strip()
    if not body:
        raise SystemExit("history entry body is empty")

    text = _read(HISTORY)
    if heading in text:
        if args.quiet:
            return 0
        print(f"history entry already exists: {heading}")
        return 0

    addition = f"\n\n{heading}\n\n{body}\n"
    _write(HISTORY, text.rstrip() + addition)
    if not args.quiet:
        print(f"appended history entry: {heading}")
    return 0


def set_status(args: argparse.Namespace) -> int:
    if args.status not in VALID_STATUSES:
        raise SystemExit(f"unknown status: {args.status}")

    text = _read(TRACKER)
    lines = text.splitlines()
    matches: list[int] = []
    for idx, line in enumerate(lines):
        if line.startswith("### ") and f": {args.item}" in line:
            matches.append(idx)

    if not matches:
        raise SystemExit(f"queue item not found: {args.item}")
    if len(matches) > 1:
        raise SystemExit(f"queue item is ambiguous: {args.item}")

    idx = matches[0]
    _, rest = lines[idx].split(": ", 1)
    lines[idx] = f"### {args.status}: {rest}"
    _write(TRACKER, "\n".join(lines) + "\n")
    if not args.quiet:
        print(f"set {args.item} to {args.status}")
    return 0


def add_note(args: argparse.Namespace) -> int:
    note = args.note.strip()
    if not note:
        raise SystemExit("note is empty")
    if args.timestamp:
        note = f"{_utc_now()} - {note}"

    text = _read(TRACKER)
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("### ") and f": {args.item}" in line:
            start = idx
            break
    if start is None:
        raise SystemExit(f"queue item not found: {args.item}")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("### ") or lines[idx].startswith("## Run History"):
            end = idx
            break

    block = [
        "",
        "Agent note:",
        "",
        "```text",
        note,
        "```",
    ]
    lines[end:end] = block
    _write(TRACKER, "\n".join(lines) + "\n")
    if not args.quiet:
        print(f"added note to {args.item}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make narrow updates to Nano AV queue/history docs."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success messages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    append = subparsers.add_parser("append-history")
    append.add_argument("heading", help="Heading id or full '### status: id' heading.")
    append.add_argument(
        "--status",
        default="completed",
        choices=sorted(VALID_STATUSES),
        help="Status to use when heading is not already a full Markdown heading.",
    )
    append.add_argument(
        "--entry-file",
        help="Markdown body to append. Reads stdin when omitted.",
    )
    append.set_defaults(func=append_history)

    status = subparsers.add_parser("set-status")
    status.add_argument("item", help="Queue item id after the status prefix.")
    status.add_argument("status", choices=sorted(VALID_STATUSES))
    status.set_defaults(func=set_status)

    note = subparsers.add_parser("add-note")
    note.add_argument("item", help="Queue item id after the status prefix.")
    note.add_argument("note")
    note.add_argument(
        "--timestamp",
        action="store_true",
        help="Prefix note with current UTC timestamp.",
    )
    note.set_defaults(func=add_note)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
