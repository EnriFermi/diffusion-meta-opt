from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def normalize_streaming_mode(value: Any) -> str:
    mode = str(value or "none").strip().lower()
    aliases = {
        "in_memory": "none",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "local_disk", "s3_bridge"}:
        raise ValueError(f"Unsupported streaming.mode='{value}'")
    return mode


@dataclass(slots=True)
class DistributedSettings:
    enabled: bool
    rank: int
    world_size: int
    shard_by: str


def resolve_distributed_settings(cfg: dict[str, Any] | None) -> DistributedSettings:
    payload = cfg or {}
    enabled = bool(payload.get("enabled", False))
    shard_by = str(payload.get("shard_by", "chunk")).lower()

    rank_env = str(payload.get("rank_env", "RANK"))
    world_env = str(payload.get("world_size_env", "WORLD_SIZE"))

    if not enabled:
        return DistributedSettings(enabled=False, rank=0, world_size=1, shard_by=shard_by)

    rank = _safe_int_env(rank_env, default=0)
    world_size = max(1, _safe_int_env(world_env, default=1))

    if rank < 0:
        rank = 0
    if rank >= world_size:
        rank = rank % world_size

    return DistributedSettings(enabled=True, rank=rank, world_size=world_size, shard_by=shard_by)


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default
