from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset.shared.streaming.backends.base import ChunkStore
from dataset.shared.streaming.chunk_format import chunk_suffix, save_chunk
from dataset.shared.types import SharedSample


@dataclass(slots=True)
class _SpoolItem:
    chunk_id: str
    local_path: Path
    meta: dict[str, Any]


class ChunkWriter:
    def __init__(
        self,
        store: ChunkStore,
        spool_dir: str | Path,
        chunk_size_samples: int,
        compression: str,
        local_max_chunks: int,
    ) -> None:
        self.store = store
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size_samples = max(1, int(chunk_size_samples))
        self.compression = str(compression or "none").lower()
        self.local_max_chunks = max(1, int(local_max_chunks))

        self.logger = logging.getLogger(self.__class__.__name__)

        self._buffer: list[SharedSample] = []
        self._sequence = 0
        self._num_chunks_written = 0
        self._num_chunks_published = 0
        self._spool_items: deque[_SpoolItem] = deque()

    def append(self, sample: SharedSample) -> None:
        self._drain_spool()

        self._buffer.append(sample)
        if len(self._buffer) >= self.chunk_size_samples:
            self.flush(force_partial=False)

    def flush(self, force_partial: bool = False) -> int:
        emitted_chunks = 0
        self._drain_spool()

        while len(self._buffer) >= self.chunk_size_samples:
            batch = self._buffer[: self.chunk_size_samples]
            if not self._enqueue_chunk(batch):
                return emitted_chunks
            del self._buffer[: self.chunk_size_samples]
            emitted_chunks += 1
            self._drain_spool()

        if force_partial and self._buffer:
            if self._enqueue_chunk(self._buffer):
                self._buffer = []
                emitted_chunks += 1
                self._drain_spool()

        return emitted_chunks

    def can_accept_more(self) -> bool:
        return self._spool_size() < self.local_max_chunks

    def pending_samples(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        self.flush(force_partial=True)
        self._drain_spool()

    def stats(self) -> dict[str, Any]:
        return {
            "pending_samples": len(self._buffer),
            "num_chunks_written": self._num_chunks_written,
            "num_chunks_published": self._num_chunks_published,
            "chunk_size_samples": self.chunk_size_samples,
            "compression": self.compression,
            "spool_dir": str(self.spool_dir),
            "spool_pending_chunks": self._spool_size(),
        }

    def _enqueue_chunk(self, samples: list[SharedSample]) -> bool:
        if not samples:
            return True

        if not self.can_accept_more():
            return False

        chunk_id = _new_chunk_id(sequence=self._sequence)
        self._sequence += 1

        local_path = self.spool_dir / f"{chunk_id}{chunk_suffix(self.compression)}"
        meta = {
            "chunk_id": chunk_id,
            "created_at": time.time(),
            "num_samples": len(samples),
        }

        save_chunk(local_path, samples=samples, meta=meta, compression=self.compression)
        self._spool_items.append(_SpoolItem(chunk_id=chunk_id, local_path=local_path, meta=meta))
        self._num_chunks_written += 1
        return True

    def _drain_spool(self) -> None:
        while self._spool_items:
            state = self.store.capacity_state()
            if not bool(state.get("can_accept", True)):
                break

            item = self._spool_items[0]
            try:
                self.store.put_ready(chunk_id=item.chunk_id, local_file=item.local_path, meta=item.meta)
            except Exception as exc:
                self.logger.warning("Failed to publish chunk %s: %s", item.chunk_id, exc)
                break

            try:
                item.local_path.unlink(missing_ok=True)
            except Exception:
                pass

            self._spool_items.popleft()
            self._num_chunks_published += 1

    def _spool_size(self) -> int:
        return len(self._spool_items)


def _new_chunk_id(sequence: int) -> str:
    return f"final_chunk_{int(time.time() * 1000)}_{sequence:08d}"
