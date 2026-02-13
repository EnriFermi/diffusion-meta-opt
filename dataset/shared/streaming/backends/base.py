from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ChunkRef:
    chunk_id: str
    uri: str
    size_bytes: int
    created_at: float
    backend_key: str


class ChunkStore(ABC):
    @abstractmethod
    def put_ready(self, chunk_id: str, local_file: str | Path, meta: dict[str, Any]) -> ChunkRef:
        raise NotImplementedError

    @abstractmethod
    def list_ready(self, limit: int | None = None) -> list[ChunkRef]:
        raise NotImplementedError

    @abstractmethod
    def fetch_to_local(self, chunk_ref: ChunkRef, target_path: str | Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def delete_ready(self, chunk_ref: ChunkRef) -> None:
        raise NotImplementedError

    @abstractmethod
    def count_ready(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def capacity_state(self) -> dict[str, Any]:
        raise NotImplementedError
