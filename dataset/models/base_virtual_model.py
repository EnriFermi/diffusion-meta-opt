from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dataset.models.types import LayerIORecord


class BaseVirtualModel(ABC):
    """Base API for model virtualization wrappers."""

    def __init__(self, cfg: Any, global_cfg: Any) -> None:
        self.cfg = cfg
        self.global_cfg = global_cfg

    @abstractmethod
    def load(self) -> None:
        """Load model and its processor/resources."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model and release memory."""

    @abstractmethod
    def run(self, batch_pil: list[Any]) -> list[LayerIORecord]:
        """Run one batch and return per-layer IO records."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return model runtime statistics."""
