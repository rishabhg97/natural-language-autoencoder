#!/usr/bin/env python3
"""AV-focused entry point for the shared Nano NLA HPO study tooling."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_ar_hpo_study import main as _shared_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return _shared_main(argv, default_task="av")


if __name__ == "__main__":
    raise SystemExit(main())
