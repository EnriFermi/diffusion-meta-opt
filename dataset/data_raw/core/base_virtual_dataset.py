from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dataset.data_raw.core.types import ImageSample, ImageSampleRef


class BaseVirtualDataset(ABC):
    """Base interface for all virtualized datasets."""

    def __init__(self, cfg: Any, global_data_root: str, seed: int) -> None:
        self.cfg = cfg
        self.global_data_root = global_data_root
        self.seed = seed

    def start(self) -> None:
        """Optional startup hook for background workers."""

    @abstractmethod
    def get_batch(self, batch_size: int) -> list[ImageSample]:
        """Return up to batch_size samples from local chunk cache."""

    @abstractmethod
    def close(self) -> None:
        """Release resources and terminate background work."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return debug stats about cache and worker state."""

    def get_batch_refs(self, batch_size: int) -> list[ImageSampleRef]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support lightweight sample refs")

    def load_refs(self, refs: list[ImageSampleRef]) -> list[ImageSample]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support loading sample refs")

    def release_refs(self, refs: list[ImageSampleRef]) -> None:
        return None
