from __future__ import annotations

from observatory.queue_after_state import upstream_outcome


def test_upstream_outcome_only_releases_complete_state() -> None:
    assert upstream_outcome({"status": "complete"}) == "ready"
    assert upstream_outcome({"status": "running"}) == "waiting"
    assert upstream_outcome({"status": "waiting_for_base"}) == "waiting"


def test_upstream_outcome_blocks_terminal_failures() -> None:
    assert upstream_outcome({"status": "failed"}) == "failed"
    assert upstream_outcome({"status": "blocked"}) == "failed"
    assert upstream_outcome({"status": "cancelled"}) == "failed"
