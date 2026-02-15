from __future__ import annotations

import unittest
from typing import Any

from dataset.shared.collector_service import _ChunkedSampleSink


class _FakeStore:
    def __init__(self, ready_chunks: int, max_ready_chunks: int) -> None:
        self.ready_chunks = int(ready_chunks)
        self.max_ready_chunks = int(max_ready_chunks)

    def capacity_state(self) -> dict[str, Any]:
        return {
            "backend": "local_disk",
            "ready_chunks": int(self.ready_chunks),
            "max_ready_chunks": int(self.max_ready_chunks),
            "low_watermark_chunks": 0,
            "can_accept": self.ready_chunks < self.max_ready_chunks,
        }

    def count_ready(self) -> int:
        return int(self.ready_chunks)


class _FakeWriter:
    def __init__(self) -> None:
        self.open_calls: list[tuple[int, int]] = []
        self.close_calls: list[bool] = []

    def append(self, sample: Any) -> None:
        return None

    def open_window(self, window_chunks: int, window_id: int) -> None:
        self.open_calls.append((int(window_chunks), int(window_id)))

    def close_window(self, flush_partial: bool = False) -> None:
        self.close_calls.append(bool(flush_partial))

    def close(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"writer": "fake"}


class TestStreamingRefillWindowLogic(unittest.TestCase):
    def test_refill_starts_after_n_read_and_stops_on_upper_bound(self) -> None:
        store = _FakeStore(ready_chunks=10, max_ready_chunks=10)
        writer = _FakeWriter()
        sink = _ChunkedSampleSink(
            mode="local_disk",
            store=store,
            writer=writer,
            refill_after_consumed_chunks=3,
            stripe_window_chunks=None,
        )

        self.assertFalse(sink.needs_fill())
        self.assertEqual(writer.open_calls, [])

        store.ready_chunks = 8
        self.assertFalse(sink.needs_fill())
        self.assertEqual(writer.open_calls, [])

        store.ready_chunks = 7
        self.assertTrue(sink.needs_fill())
        self.assertEqual(writer.open_calls, [(3, 0)])

        store.ready_chunks = 9
        self.assertTrue(sink.needs_fill())

        store.ready_chunks = 10
        self.assertFalse(sink.needs_fill())
        self.assertEqual(writer.close_calls, [False])

        store.ready_chunks = 7
        self.assertTrue(sink.needs_fill())
        self.assertEqual(writer.open_calls, [(3, 0), (3, 1)])


if __name__ == "__main__":
    unittest.main()
