from __future__ import annotations

import logging
from typing import Iterator

import torch

from dataset.shared.collector_service import CollectorService
from dataset.shared.types import SharedSample


class SharedModelDataset(torch.utils.data.IterableDataset):
    """Training-facing dataset that yields cached SharedSample objects."""

    def __init__(self, collector: CollectorService) -> None:
        super().__init__()
        self.collector = collector
        self.logger = logging.getLogger(self.__class__.__name__)

    def __iter__(self) -> Iterator[SharedSample]:
        while True:
            sample = self.collector.cache.get(block=True)
            yield sample

    def maybe_collect(self, step_idx: int):
        """Interleaved-mode hook for training loop."""
        return self.collector.maybe_collect(step_idx)

    def cache_size(self) -> int:
        return self.collector.cache.size()
