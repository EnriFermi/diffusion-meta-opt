from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from dataset.shared.streaming.backends.base import ChunkRef, ChunkStore
from dataset.shared.streaming.chunk_format import load_chunk
from dataset.shared.streaming.config import DistributedSettings, resolve_distributed_settings
from dataset.shared.types import SharedSample


@dataclass(slots=True)
class _LocalChunk:
    ref: ChunkRef
    local_path: Path
    remote_deleted: bool


class ChunkReader:
    def __init__(
        self,
        store: ChunkStore,
        cache_dir: str | Path,
        prefetch_max_chunks: int,
        delete_remote_after: str,
        distributed_cfg: dict[str, Any] | None,
        randomize_chunk_order: bool = False,
        randomize_within_chunk: bool = True,
        random_seed: int | None = None,
    ) -> None:
        self.store = store
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.prefetch_max_chunks = max(1, int(prefetch_max_chunks))
        self.delete_remote_after = str(delete_remote_after or "consume").lower()
        if self.delete_remote_after not in {"consume", "download"}:
            raise ValueError("streaming.consumer.delete_remote_after must be 'consume' or 'download'")

        self.distributed: DistributedSettings = resolve_distributed_settings(distributed_cfg)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.randomize_chunk_order = bool(randomize_chunk_order)
        self.randomize_within_chunk = bool(randomize_within_chunk)
        self._rng = random.Random(random_seed if random_seed is not None else time.time_ns())

        self._pending_chunks: deque[_LocalChunk] = deque()
        self._known_chunk_ids: set[str] = set()

        self._active_chunk: _LocalChunk | None = None
        self._active_samples: list[SharedSample] = []
        self._active_index = 0
        self._active_order: list[int] = []

    def next_sample(self, block: bool = True, timeout: float | None = None) -> SharedSample:
        deadline = None
        if block and timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            sample = self.try_next_sample()
            if sample is not None:
                return sample

            if not block:
                raise Empty("No ready chunk sample")

            if deadline is not None and time.monotonic() >= deadline:
                raise Empty("Timed out waiting for streaming chunk")

            time.sleep(0.1)

    def try_next_sample(self) -> SharedSample | None:
        if self._active_chunk is None or self._active_index >= len(self._active_samples):
            self._rotate_chunk()

        if self._active_chunk is None:
            return None
        if not self._active_order:
            self._finalize_active_chunk()
            self._prefetch_once()
            return None

        read_index = self._active_order[self._active_index]
        sample = self._active_samples[read_index]
        self._active_index += 1

        if self._active_index >= len(self._active_samples):
            self._finalize_active_chunk()
            self._prefetch_once()

        return sample

    def size_metric(self) -> int:
        return self.store.count_ready()

    def close(self) -> None:
        self._finalize_active_chunk(delete_remote=False)
        while self._pending_chunks:
            item = self._pending_chunks.popleft()
            try:
                item.local_path.unlink(missing_ok=True)
            except Exception:
                pass

    def debug_snapshot(self, preview: int = 5) -> dict[str, Any]:
        preview_int = max(1, int(preview))
        pending_items = list(self._pending_chunks)
        pending_chunk_ids = [item.ref.chunk_id for item in pending_items[:preview_int]]
        active_chunk_id = self._active_chunk.ref.chunk_id if self._active_chunk is not None else None

        active_total = len(self._active_samples)
        active_remaining = max(0, active_total - int(self._active_index))

        local_cache_files = [
            path.name
            for path in self.cache_dir.iterdir()
            if path.is_file() and not path.name.endswith(".tmp")
        ]
        local_cache_files.sort()

        return {
            "pending_chunks_count": len(pending_items),
            "pending_chunk_ids_head": pending_chunk_ids,
            "active_chunk_id": active_chunk_id,
            "active_samples_total": int(active_total),
            "active_samples_remaining": int(active_remaining),
            "known_chunk_ids_count": int(len(self._known_chunk_ids)),
            "local_cache_files_count": int(len(local_cache_files)),
            "local_cache_files_head": local_cache_files[:preview_int],
        }

    def _rotate_chunk(self) -> None:
        self._finalize_active_chunk()
        self._prefetch_once()

        if not self._pending_chunks:
            self._active_chunk = None
            self._active_samples = []
            self._active_index = 0
            return

        self._active_chunk = self._pending_chunks.popleft()
        samples, _ = load_chunk(self._active_chunk.local_path)
        self._active_samples = samples
        self._active_index = 0
        self._active_order = list(range(len(self._active_samples)))
        if self.randomize_within_chunk and len(self._active_order) > 1:
            self._rng.shuffle(self._active_order)

    def _prefetch_once(self) -> None:
        while len(self._pending_chunks) < self.prefetch_max_chunks:
            refs = self.store.list_ready(limit=self.prefetch_max_chunks * 4)
            refs = [ref for ref in refs if self._eligible_ref(ref)]

            if not refs:
                return

            if self.randomize_chunk_order:
                ref = self._rng.choice(refs)
            else:
                ref = refs[0]
            self._known_chunk_ids.add(ref.chunk_id)

            target = self.cache_dir / _chunk_filename(ref)
            self.store.fetch_to_local(ref, target)

            remote_deleted = False
            if self.delete_remote_after == "download":
                self.store.delete_ready(ref)
                remote_deleted = True

            self._pending_chunks.append(_LocalChunk(ref=ref, local_path=target, remote_deleted=remote_deleted))

    def _eligible_ref(self, ref: ChunkRef) -> bool:
        if ref.chunk_id in self._known_chunk_ids:
            return False

        if not self.distributed.enabled:
            return True

        if self.distributed.shard_by != "chunk":
            return True

        assigned = assign_chunk_rank(ref.chunk_id, self.distributed.world_size)
        return assigned == self.distributed.rank

    def _finalize_active_chunk(self, delete_remote: bool = True) -> None:
        if self._active_chunk is None:
            return

        chunk = self._active_chunk
        if delete_remote and self.delete_remote_after == "consume" and not chunk.remote_deleted:
            self.store.delete_ready(chunk.ref)

        try:
            chunk.local_path.unlink(missing_ok=True)
        except Exception:
            pass

        self._active_chunk = None
        self._active_samples = []
        self._active_index = 0
        self._active_order = []


def assign_chunk_rank(chunk_id: str, world_size: int) -> int:
    if world_size <= 1:
        return 0

    digest = hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % int(world_size)


def _chunk_filename(ref: ChunkRef) -> str:
    if ref.backend_key.startswith("s3://"):
        name = ref.backend_key.rsplit("/", 1)[-1]
    else:
        name = Path(ref.backend_key).name
    if name:
        return name
    return f"{ref.chunk_id}.pt"
