from __future__ import annotations

import logging
from queue import Empty
from typing import Iterator

import torch

from dataset.shared.collector_service import CollectorService
from dataset.shared.streaming.chunk_reader import ChunkReader
from dataset.shared.types import SharedSample


class SharedModelDataset(torch.utils.data.IterableDataset):
    """Training-facing dataset that yields SharedSample from memory cache or chunk stream."""

    def __init__(self, collector: CollectorService) -> None:
        super().__init__()
        self.collector = collector
        self.logger = logging.getLogger(self.__class__.__name__)
        self._reader: ChunkReader | None = None

    def __iter__(self) -> Iterator[SharedSample]:
        while True:
            if self.collector.streaming_mode == "none":
                if self.collector.cache is None:
                    raise RuntimeError("Collector has no in-memory cache for streaming.mode=none")
                sample = self.collector.cache.get(block=True)
                yield sample
                continue

            reader = self._ensure_reader()
            try:
                sample = reader.next_sample(block=True, timeout=1.0)
            except Empty:
                continue
            yield sample

    def maybe_collect(self, step_idx: int):
        """Interleaved-mode hook for training loop."""
        return self.collector.maybe_collect(step_idx)

    def try_next_sample(self) -> SharedSample | None:
        if self.collector.streaming_mode == "none":
            if self.collector.cache is None:
                return None
            return self.collector.cache.try_get()

        reader = self._ensure_reader()
        return reader.try_next_sample()

    def cache_size(self) -> int:
        return self.collector.cache_size()

    def debug_snapshot(self, preview: int = 5) -> dict[str, object]:
        payload: dict[str, object] = {
            "streaming_mode": self.collector.streaming_mode,
            "cache_metric": self.cache_size(),
        }

        if self.collector.streaming_mode != "none" and self._reader is not None:
            payload["reader"] = self._reader.debug_snapshot(preview=preview)
        return payload

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _ensure_reader(self) -> ChunkReader:
        if self._reader is not None:
            return self._reader

        reader = self.collector.create_chunk_reader()
        if reader is None:
            raise RuntimeError(
                f"Collector streaming_mode='{self.collector.streaming_mode}' does not provide a chunk reader"
            )
        self._reader = reader
        return reader
