"""Compatibility launcher for commands moved into the Observatory package."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


def run(module_name: str) -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    return int(module.main())
