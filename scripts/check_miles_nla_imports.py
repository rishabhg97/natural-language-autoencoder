#!/usr/bin/env python3
"""Fail-fast import gate for Miles plus Qwen NLA extension points."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


REQUIRED_SYMBOLS = (
    ("miles", None),
    ("train", "train"),
    ("miles.ray.rollout", "RolloutManager"),
    ("nla.train_actor", "NLAFSDPActor"),
    ("nla.rollout.sft_actor", "generate_rollout"),
    ("nla.injection", "inject_at_marked_positions"),
)


def check_imports(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    checks = []
    for module_name, symbol_name in REQUIRED_SYMBOLS:
        target = module_name if symbol_name is None else f"{module_name}.{symbol_name}"
        try:
            module = import_module(module_name)
            if symbol_name is not None:
                getattr(module, symbol_name)
            checks.append(
                {
                    "target": target,
                    "status": "ok",
                    "module_file": str(getattr(module, "__file__", "")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - import gate should report all failures.
            checks.append(
                {
                    "target": target,
                    "status": "missing",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"ok": all(check["status"] == "ok" for check in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nla-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "natural_language_autoencoders",
        help="Source root containing the nla package, added to sys.path before import checks.",
    )
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    if args.nla_root.is_dir():
        sys.path.insert(0, str(args.nla_root))

    report = check_imports()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
