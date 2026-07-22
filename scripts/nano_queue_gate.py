#!/usr/bin/env python3
"""Gate one Nano experiment queue item on evidence from another queue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_queue(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"queue YAML must contain a mapping: {path}")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"queue YAML must contain an items list: {path}")
    return data


def write_queue(path: str | Path, queue_doc: dict[str, Any]) -> None:
    destination = Path(path)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(destination)


def find_item(queue_doc: dict[str, Any], item_name: str) -> dict[str, Any]:
    for item in queue_doc.get("items", []):
        if isinstance(item, dict) and item.get("name") == item_name:
            return item
    raise ValueError(f"queue item not found: {item_name}")


def _path_from_field(item: dict[str, Any], field: str) -> Path | None:
    value = item.get(field)
    if value in {None, ""}:
        return None
    return Path(str(value))


def _json_path_value(data: Any, dotted_path: str) -> Any:
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _json_bool_reasons(required_json_bools: list[tuple[Path, str]]) -> list[str]:
    reasons: list[str] = []
    for path, dotted_path in required_json_bools:
        if not path.exists():
            reasons.append(f"missing required JSON path: {path}")
            continue
        try:
            data = json.loads(path.read_text())
            value = _json_path_value(data, dotted_path)
        except Exception as exc:
            reasons.append(f"required JSON bool {dotted_path} unreadable in {path}: {exc}")
            continue
        if value is not True:
            reasons.append(f"required JSON bool {dotted_path} is {value!r}, not true in {path}")
    return reasons


def readiness_reasons(
    *,
    dependency_item: dict[str, Any],
    required_fields: list[str],
    required_status: str,
    extra_required_paths: list[Path],
    required_json_bools: list[tuple[Path, str]] | None = None,
) -> list[str]:
    reasons: list[str] = []
    status = str(dependency_item.get("status", "pending"))
    if status != required_status:
        reasons.append(f"dependency status is {status}, not {required_status}")
    for field in required_fields:
        path = _path_from_field(dependency_item, field)
        if path is None:
            reasons.append(f"dependency field {field} is missing")
        elif not path.exists():
            reasons.append(f"missing dependency path {field}: {path}")
    for path in extra_required_paths:
        if not path.exists():
            reasons.append(f"missing required path: {path}")
    reasons.extend(_json_bool_reasons(required_json_bools or []))
    return reasons


def unblock_when_ready(
    *,
    dependency_queue: str | Path,
    dependency_item_name: str,
    target_queue: str | Path,
    target_item_name: str,
    required_fields: list[str],
    dry_run: bool,
    now: str | None = None,
    required_status: str = "complete",
    extra_required_paths: list[str | Path] | None = None,
    required_json_bools: list[tuple[str | Path, str]] | None = None,
) -> dict[str, Any]:
    dependency_queue_path = Path(dependency_queue)
    target_queue_path = Path(target_queue)
    dependency_doc = load_queue(dependency_queue_path)
    target_doc = load_queue(target_queue_path)
    dependency_item = find_item(dependency_doc, dependency_item_name)
    target_item = find_item(target_doc, target_item_name)
    required_paths = [Path(path) for path in (extra_required_paths or [])]
    json_bools = [(Path(path), dotted_path) for path, dotted_path in (required_json_bools or [])]
    reasons = readiness_reasons(
        dependency_item=dependency_item,
        required_fields=required_fields,
        required_status=required_status,
        extra_required_paths=required_paths,
        required_json_bools=json_bools,
    )
    ready = not reasons
    result: dict[str, Any] = {
        "ready": ready,
        "changed": False,
        "dependency_queue": str(dependency_queue_path),
        "dependency_item": dependency_item_name,
        "target_queue": str(target_queue_path),
        "target_item": target_item_name,
        "target_status": target_item.get("status", "pending"),
        "reasons": reasons,
    }
    if not ready:
        return result
    if target_item.get("status") != "blocked":
        result["reasons"] = [f"target status is {target_item.get('status')}, not blocked"]
        return result
    if dry_run:
        result["changed"] = False
        result["would_set_status"] = "pending"
        return result

    target_item["previous_status"] = target_item.get("status")
    target_item["status"] = "pending"
    target_item["unblocked_at"] = now or utc_now()
    target_item["unblock_dependency_queue"] = str(dependency_queue_path)
    target_item["unblock_dependency_item"] = dependency_item_name
    target_item["unblock_required_fields"] = list(required_fields)
    write_queue(target_queue_path, target_doc)
    result["changed"] = True
    result["target_status"] = "pending"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-queue", required=True, type=Path)
    parser.add_argument("--dependency-item", required=True)
    parser.add_argument("--target-queue", required=True, type=Path)
    parser.add_argument("--target-item", required=True)
    parser.add_argument("--required-field", action="append", default=[])
    parser.add_argument("--required-status", default="complete")
    parser.add_argument("--required-path", action="append", default=[])
    parser.add_argument(
        "--required-json-bool",
        action="append",
        default=[],
        metavar="PATH:DOT.PATH",
        help="Require a JSON file dotted path to be exactly true, e.g. report.json:gate.passed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    required_json_bools: list[tuple[Path, str]] = []
    for value in args.required_json_bool:
        raw = str(value)
        if ":" not in raw:
            raise SystemExit(f"--required-json-bool must be PATH:DOT.PATH, got {raw!r}")
        path, dotted_path = raw.rsplit(":", 1)
        required_json_bools.append((Path(path), dotted_path))
    result = unblock_when_ready(
        dependency_queue=args.dependency_queue,
        dependency_item_name=args.dependency_item,
        target_queue=args.target_queue,
        target_item_name=args.target_item,
        required_fields=[str(field) for field in args.required_field],
        required_status=str(args.required_status),
        extra_required_paths=[Path(path) for path in args.required_path],
        required_json_bools=required_json_bools,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
