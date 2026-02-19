from __future__ import annotations

import unittest
from queue import Empty

import torch

from dataset.shared.shared_dataset import SharedModelDataset
from dataset.shared.types import SharedSample


class _FlakyCache:
    def __init__(self, sample: SharedSample) -> None:
        self._sample = sample
        self._calls = 0

    def get(self, block: bool = True, timeout: float | None = None) -> SharedSample:
        self._calls += 1
        if self._calls == 1:
            raise Empty()
        return self._sample


class _AlwaysEmptyCache:
    def get(self, block: bool = True, timeout: float | None = None) -> SharedSample:
        raise Empty()


class _FailingReader:
    def next_sample(self, block: bool = True, timeout: float | None = None):
        raise Empty()


class _HealthyCollectorInMemory:
    streaming_mode = "none"

    def __init__(self, sample: SharedSample) -> None:
        self.cache = _FlakyCache(sample)
        self.health_checks = 0

    def assert_healthy(self) -> None:
        self.health_checks += 1


class _UnhealthyCollectorInMemory:
    streaming_mode = "none"

    def __init__(self) -> None:
        self.cache = _AlwaysEmptyCache()

    def assert_healthy(self) -> None:
        raise RuntimeError("collector process exited")


class _UnhealthyCollectorStreaming:
    streaming_mode = "local_disk"
    cache = None

    def __init__(self) -> None:
        self._reader = _FailingReader()

    def assert_healthy(self) -> None:
        raise RuntimeError("collector process exited")

    def create_chunk_reader(self):
        return self._reader


class TestSharedModelDatasetHealth(unittest.TestCase):
    @staticmethod
    def _sample() -> SharedSample:
        tensor = torch.zeros((1, 1), dtype=torch.float32)
        return SharedSample(
            model_name="m",
            layer_name="l",
            weight=tensor,
            x=tensor,
            y=tensor,
            meta={},
        )

    def test_in_memory_recovers_after_temporary_empty_cache(self) -> None:
        collector = _HealthyCollectorInMemory(sample=self._sample())
        dataset = SharedModelDataset(collector=collector)  # type: ignore[arg-type]
        item = next(iter(dataset))
        self.assertEqual(item.model_name, "m")
        self.assertEqual(collector.health_checks, 1)

    def test_in_memory_raises_when_collector_unhealthy(self) -> None:
        collector = _UnhealthyCollectorInMemory()
        dataset = SharedModelDataset(collector=collector)  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "collector process exited"):
            next(iter(dataset))

    def test_streaming_raises_when_collector_unhealthy(self) -> None:
        collector = _UnhealthyCollectorStreaming()
        dataset = SharedModelDataset(collector=collector)  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "collector process exited"):
            next(iter(dataset))


if __name__ == "__main__":
    unittest.main()
