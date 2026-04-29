from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class PrefetchedItem(Generic[T]):
    value: T
    wait_time_s: float


@dataclass(slots=True)
class _QueueItem(Generic[T]):
    kind: str
    value: T | None = None
    error: BaseException | None = None
    traceback_text: str | None = None


class BackgroundPrefetchError(RuntimeError):
    pass


class BackgroundPrefetcher(Generic[T]):
    def __init__(
        self,
        *,
        build_fn: Callable[[], T],
        queue_size: int,
        name: str = "background_prefetch",
    ) -> None:
        if queue_size <= 0:
            raise ValueError(f"queue_size must be > 0, got {queue_size}")
        self._build_fn = build_fn
        self._name = str(name).strip() or "background_prefetch"
        self._queue: queue.Queue[_QueueItem[T]] = queue.Queue(maxsize=int(queue_size))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=self._name,
            daemon=True,
        )
        self._started = False

    def __enter__(self) -> BackgroundPrefetcher[T]:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def get(self) -> PrefetchedItem[T]:
        if not self._started:
            self.start()
        wait_t0 = time.perf_counter()
        item = self._queue.get()
        wait_time_s = time.perf_counter() - wait_t0
        if item.kind == "value":
            return PrefetchedItem(value=item.value, wait_time_s=float(wait_time_s))
        if item.kind == "end":
            raise StopIteration
        if item.kind == "error":
            message = f"{self._name} failed in background producer"
            if item.traceback_text:
                message = f"{message}\n{item.traceback_text}"
            raise BackgroundPrefetchError(message) from item.error
        raise RuntimeError(f"Unknown prefetch queue item kind: {item.kind!r}")

    def close(self) -> None:
        self._stop_event.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @property
    def qsize(self) -> int:
        return int(self._queue.qsize())

    def _put(self, item: _QueueItem[T]) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    value = self._build_fn()
                except StopIteration:
                    self._put(_QueueItem(kind="end"))
                    return
                except BaseException as exc:  # pragma: no cover - producer crash path
                    self._put(
                        _QueueItem(
                            kind="error",
                            error=exc,
                            traceback_text=traceback.format_exc(),
                        )
                    )
                    return
                self._put(_QueueItem(kind="value", value=value))
        finally:
            self._stop_event.set()
