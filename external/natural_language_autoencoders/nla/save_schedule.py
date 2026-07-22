"""Config-driven checkpoint scheduling for Miles training loops."""

from __future__ import annotations

from functools import lru_cache
import os

from miles.utils.misc import should_run_periodic_action


@lru_cache(maxsize=None)
def parse_save_iterations(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated, strictly increasing set of completed updates."""

    values: list[int] = []
    for part in raw.split(","):
        value_text = part.strip()
        if not value_text:
            continue
        try:
            value = int(value_text)
        except ValueError as exc:
            raise ValueError(
                f"NLA_SAVE_ITERATIONS must contain integers, got {value_text!r}"
            ) from exc
        if value <= 0:
            raise ValueError("NLA_SAVE_ITERATIONS values must be positive")
        values.append(value)
    if not values:
        raise ValueError("NLA_SAVE_ITERATIONS must not be empty")
    if values != sorted(set(values)):
        raise ValueError(
            "NLA_SAVE_ITERATIONS must be unique and strictly increasing"
        )
    return tuple(values)


def should_save_rollout(
    rollout_id: int,
    interval: int | None,
    num_rollout_per_epoch: int | None = None,
    num_rollout: int | None = None,
) -> bool:
    """Use an explicit update schedule when configured, otherwise Miles cadence."""

    raw = os.environ.get("NLA_SAVE_ITERATIONS", "").strip()
    if not raw:
        return should_run_periodic_action(
            rollout_id,
            interval,
            num_rollout_per_epoch,
            num_rollout,
        )
    iterations = parse_save_iterations(raw)
    if num_rollout is not None and iterations[-1] != int(num_rollout):
        raise ValueError(
            "NLA_SAVE_ITERATIONS must include the final configured rollout "
            f"({num_rollout}), got {iterations[-1]}"
        )
    return rollout_id + 1 in iterations
