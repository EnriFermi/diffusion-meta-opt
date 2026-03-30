from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
from typing import Any

try:
    import torch.multiprocessing as torch_mp
except Exception:  # pragma: no cover - torch may be unavailable in lightweight environments
    torch_mp = None


def _configure_torch_queue_sharing() -> None:
    if torch_mp is None:
        return
    try:
        if torch_mp.get_sharing_strategy() != "file_system":
            torch_mp.set_sharing_strategy("file_system")
    except Exception:
        pass


_configure_torch_queue_sharing()

from dataset.shared.shm_transport import SharedSampleRef, cleanup_shared_sample_ref, restore_shared_sample, share_shared_sample
from dataset.shared.types import SharedSample


class SharedSampleCache:
    """Process-safe bounded cache with hysteresis thresholds."""

    def __init__(
        self,
        max_items: int,
        fill_target: int,
        low_watermark: int,
        ctx: mp.context.BaseContext | None = None,
    ) -> None:
        if max_items <= 0:
            raise ValueError("cache.max_items must be > 0")

        self.max_items = int(max_items)
        self.fill_target = min(int(fill_target), self.max_items)
        self.low_watermark = min(int(low_watermark), self.fill_target)

        self._ctx = ctx or mp.get_context("spawn")
        self._queue: Any = self._ctx.Queue(maxsize=self.max_items)
        self._size = self._ctx.Value("i", 0)
        self._lock = self._ctx.Lock()

    def put(self, sample: Any, block: bool = True, timeout: float | None = None) -> None:
        payload = share_shared_sample(sample) if isinstance(sample, SharedSample) else sample
        pushed = False
        try:
            self._queue.put(payload, block=block, timeout=timeout)
            pushed = True
            with self._lock:
                self._size.value += 1
        finally:
            if not pushed and isinstance(payload, SharedSampleRef):
                cleanup_shared_sample_ref(payload)

    def put_many(self, samples: list[Any], block: bool = True, timeout: float | None = None) -> int:
        pushed = 0
        for sample in samples:
            try:
                self.put(sample, block=block, timeout=timeout)
            except Full:
                break
            pushed += 1
        return pushed

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        item = self._queue.get(block=block, timeout=timeout)
        with self._lock:
            self._size.value = max(0, self._size.value - 1)
        if isinstance(item, SharedSampleRef):
            return restore_shared_sample(item, release=True)
        return item

    def get_nowait(self) -> Any:
        return self.get(block=False)

    def try_get(self) -> Any | None:
        try:
            return self.get(block=False)
        except Empty:
            return None

    def size(self) -> int:
        with self._lock:
            return int(self._size.value)

    def needs_fill(self) -> bool:
        return self.size() < self.fill_target

    def is_low(self) -> bool:
        return self.size() < self.low_watermark

    def empty(self) -> bool:
        return self.size() <= 0

    def full(self) -> bool:
        return self.size() >= self.max_items

    def close(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
            if isinstance(item, SharedSampleRef):
                cleanup_shared_sample_ref(item)
        try:
            self._queue.close()
        except Exception:
            pass
