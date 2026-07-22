"""Shared queue status constants for Nano experiment runners."""

from __future__ import annotations

from typing import Any


PENDING_STATUS = "pending"
ACTIVE_HPO_STATUSES = {"training", "eval_running"}
ACTIVE_LAYER_STATUSES = {"running"}
TERMINAL_STATUSES = {"complete", "failed", "cancelled", "blocked", "blocked_missing_dataset"}
VALID_HPO_STATUSES = {PENDING_STATUS, *ACTIVE_HPO_STATUSES, *TERMINAL_STATUSES}
VALID_LAYER_STATUSES = {PENDING_STATUS, *ACTIVE_LAYER_STATUSES, *TERMINAL_STATUSES}


def status_counts(items: list[dict[str, Any]], valid_statuses: set[str]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(valid_statuses)}
    for item in items:
        counts[str(item.get("status", PENDING_STATUS))] = counts.get(str(item.get("status")), 0) + 1
    return counts
