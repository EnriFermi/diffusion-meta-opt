"""Step-addressed replayable sampling shared by both codec arms."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class ResettableEpochSampler(Sampler[int]):
    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.base_seed = int(seed)
        self.epoch = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.base_seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.base_seed + self.epoch)
        self.epoch += 1
        yield from torch.randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        return self.size
