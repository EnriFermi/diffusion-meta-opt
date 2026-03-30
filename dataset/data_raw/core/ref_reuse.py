from __future__ import annotations

from typing import Any, Callable


def resolve_max_distinct_consumers_per_sample(
    *,
    configured_max: int,
    requested_max: int | None,
    compatible_consumers: tuple[str, ...],
) -> int:
    limit = int(configured_max if requested_max is None else requested_max)
    limit = max(1, limit)
    if compatible_consumers:
        limit = min(limit, len(compatible_consumers))
    return max(1, limit)


def select_reusable_item_index(
    *,
    items: list[dict[str, Any]],
    consumer_histories: list[set[str]],
    retired_indices: set[int],
    consumer_id: str,
    max_distinct_consumers_per_sample: int,
    file_exists: Callable[[str], bool],
) -> int | None:
    for index, entry in enumerate(items):
        if index in retired_indices:
            continue

        history = consumer_histories[index]
        if len(history) >= max_distinct_consumers_per_sample:
            continue
        if consumer_id in history:
            continue

        file_name = entry.get("file")
        if not file_name or not file_exists(str(file_name)):
            retired_indices.add(index)
            continue

        return index

    return None


def retire_stale_reusable_items(
    *,
    consumer_histories: list[set[str]],
    first_issued_at: list[float | None],
    retired_indices: set[int],
    now_s: float,
    max_pending_reuse_s: float | None,
    max_distinct_consumers_per_sample: int,
) -> None:
    if max_pending_reuse_s is None:
        return
    threshold_s = float(max_pending_reuse_s)
    if threshold_s <= 0:
        return

    for index, history in enumerate(consumer_histories):
        if index in retired_indices:
            continue
        if not history:
            continue
        if len(history) >= max_distinct_consumers_per_sample:
            continue
        issued_at = first_issued_at[index]
        if issued_at is None:
            continue
        if (float(now_s) - float(issued_at)) >= threshold_s:
            retired_indices.add(index)


def chunk_fully_consumed(
    *,
    num_items: int,
    consumer_histories: list[set[str]],
    retired_indices: set[int],
    max_distinct_consumers_per_sample: int,
) -> bool:
    for index in range(max(0, int(num_items))):
        if index in retired_indices:
            continue
        if len(consumer_histories[index]) < max_distinct_consumers_per_sample:
            return False
    return True
