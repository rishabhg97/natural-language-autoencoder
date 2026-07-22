"""Deterministic row balancing for online NLA critic training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT_KEY = "_nla_critic_repartition_report"
REPLICATED_GLOBAL_LIST_FIELDS = frozenset({"raw_reward"})


class CriticRepartitionError(ValueError):
    """Raised when rollout shards cannot be safely balanced for the critic."""


def _sample_fields(partition: Mapping[str, Any]) -> set[str]:
    return {
        key
        for key, value in partition.items()
        if key
        not in {
            "partition",
            "total_lengths",
            REPORT_KEY,
            *REPLICATED_GLOBAL_LIST_FIELDS,
        }
        and isinstance(value, list)
    }


def _validate_replicated_global_fields(
    actor_partitions: Sequence[Mapping[str, Any]],
    total_lengths: Sequence[Any],
) -> tuple[str, ...]:
    """Validate Miles fields replicated before train-side row selection."""

    present_fields: list[str] = []
    for key in sorted(REPLICATED_GLOBAL_LIST_FIELDS):
        present = [key in partition for partition in actor_partitions]
        if any(present) and not all(present):
            raise CriticRepartitionError(
                f"replicated global field {key!r} is missing from some actor partitions"
            )
        if not any(present):
            continue

        expected = actor_partitions[0][key]
        if not isinstance(expected, list) or len(expected) != len(total_lengths):
            actual_length = len(expected) if isinstance(expected, list) else None
            raise CriticRepartitionError(
                f"replicated global field {key!r} has {actual_length} rows; "
                f"expected {len(total_lengths)}"
            )
        for actor_rank, partition in enumerate(actor_partitions[1:], start=1):
            value = partition[key]
            if not isinstance(value, list) or len(value) != len(total_lengths):
                actual_length = len(value) if isinstance(value, list) else None
                raise CriticRepartitionError(
                    f"actor partition {actor_rank} replicated global field {key!r} "
                    f"has {actual_length} rows; expected {len(total_lengths)}"
                )
            if value != expected:
                raise CriticRepartitionError(
                    f"actor partition {actor_rank} has inconsistent replicated "
                    f"global field {key!r}"
                )
        present_fields.append(key)
    return tuple(present_fields)


def balance_critic_partition(
    actor_partitions: Sequence[Mapping[str, Any]],
    *,
    critic_rank: int,
    critic_dp: int,
    alignment: int = 1,
    required_multimodal_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, int | float | str]]:
    """Return one equal-sized critic shard assembled from all actor shards.

    Actor rollout data is partitioned by actor DP. Assigning whole actor shards
    to a smaller critic DP creates unequal local batches, after which FSDP must
    truncate every rank to the smallest shard. This function instead rebuilds
    the global row order, removes rows unusable by the critic, aligns the global
    count to ``critic_dp * alignment``, and distributes rows round-robin.

    Values are treated as opaque objects; this module deliberately has no Ray,
    Torch, or model dependency so the batching contract is easy to unit test.
    """

    if not actor_partitions:
        raise CriticRepartitionError("actor_partitions must not be empty")
    if critic_dp <= 0:
        raise CriticRepartitionError("critic_dp must be positive")
    if critic_rank < 0 or critic_rank >= critic_dp:
        raise CriticRepartitionError(
            f"critic_rank={critic_rank} must be in [0, {critic_dp})"
        )
    if alignment <= 0:
        raise CriticRepartitionError("alignment must be positive")

    first = actor_partitions[0]
    if "total_lengths" not in first:
        raise CriticRepartitionError("actor partition is missing total_lengths")
    total_lengths = list(first["total_lengths"])
    replicated_global_fields = _validate_replicated_global_fields(
        actor_partitions,
        total_lengths,
    )
    fields = _sample_fields(first)
    rows: list[tuple[int, dict[str, Any]]] = []

    for actor_rank, partition in enumerate(actor_partitions):
        if list(partition.get("total_lengths", [])) != total_lengths:
            raise CriticRepartitionError(
                f"actor partition {actor_rank} has inconsistent total_lengths"
            )
        indices = list(partition.get("partition", []))
        current_fields = _sample_fields(partition)
        if current_fields != fields:
            raise CriticRepartitionError(
                f"actor partition {actor_rank} has inconsistent sample fields: "
                f"expected={sorted(fields)} actual={sorted(current_fields)}"
            )
        for key in fields:
            value = partition[key]
            if len(value) != len(indices):
                raise CriticRepartitionError(
                    f"actor partition {actor_rank} field {key!r} has "
                    f"{len(value)} rows for {len(indices)} partition indices"
                )
        for local_index, global_index in enumerate(indices):
            global_index = int(global_index)
            if global_index < 0 or global_index >= len(total_lengths):
                raise CriticRepartitionError(
                    f"actor partition {actor_rank} has out-of-range row "
                    f"index {global_index}"
                )
            rows.append(
                (
                    global_index,
                    {key: partition[key][local_index] for key in fields},
                )
            )

    rows.sort(key=lambda item: item[0])
    row_indices = [index for index, _ in rows]
    if len(row_indices) != len(set(row_indices)):
        raise CriticRepartitionError("actor partitions contain duplicate row indices")
    if len(rows) != len(total_lengths):
        raise CriticRepartitionError(
            f"actor partitions cover {len(rows)} rows but total_lengths has "
            f"{len(total_lengths)} rows"
        )

    usable_rows = rows
    if required_multimodal_key is not None:
        usable_rows = []
        for row in rows:
            multimodal = row[1].get("multimodal_train_inputs")
            if isinstance(multimodal, Mapping) and required_multimodal_key in multimodal:
                usable_rows.append(row)

    batch_alignment = critic_dp * alignment
    retained_count = (len(usable_rows) // batch_alignment) * batch_alignment
    if retained_count <= 0:
        raise CriticRepartitionError(
            "no critic batch remains after parse filtering and DP/microbatch "
            f"alignment: usable={len(usable_rows)} alignment={batch_alignment}"
        )
    retained_rows = usable_rows[:retained_count]
    local_rows = retained_rows[critic_rank::critic_dp]
    expected_local = retained_count // critic_dp
    if len(local_rows) != expected_local:
        raise CriticRepartitionError(
            f"balanced critic shard has {len(local_rows)} rows, expected "
            f"{expected_local}"
        )

    output: dict[str, Any] = {
        "total_lengths": total_lengths,
        "partition": [index for index, _ in local_rows],
    }
    for key in fields:
        output[key] = [row[key] for _, row in local_rows]
    for key, value in first.items():
        if key not in output and key not in fields and key != REPORT_KEY:
            output[key] = value

    usable_count = len(usable_rows)
    generated_count = len(rows)
    report: dict[str, int | float | str] = {
        "mode": "balanced_rows_v1",
        "actor_partitions": len(actor_partitions),
        "critic_dp": critic_dp,
        "alignment": alignment,
        "generated_samples": generated_count,
        "usable_samples": usable_count,
        "retained_samples": retained_count,
        "local_samples": expected_local,
        "dropped_unusable_samples": generated_count - usable_count,
        "dropped_alignment_samples": usable_count - retained_count,
        "usable_fraction": usable_count / generated_count,
        "retained_fraction_of_usable": retained_count / usable_count,
        "retained_fraction_of_generated": retained_count / generated_count,
        "replicated_global_fields": ",".join(replicated_global_fields),
    }
    output[REPORT_KEY] = report
    return output, report


def require_minimum_retained_fraction(
    report: Mapping[str, Any], minimum: float
) -> None:
    """Fail when balanced critic training retains too few usable rows."""

    if minimum < 0.0 or minimum > 1.0:
        raise CriticRepartitionError(
            f"minimum retained fraction must be in [0, 1], got {minimum}"
        )
    actual = float(report["retained_fraction_of_usable"])
    if actual < minimum:
        raise CriticRepartitionError(
            "critic retained fraction is below the configured minimum: "
            f"actual={actual:.6f} minimum={minimum:.6f} "
            f"usable={report['usable_samples']} retained={report['retained_samples']}"
        )
