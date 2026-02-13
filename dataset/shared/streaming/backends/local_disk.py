from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from dataset.shared.streaming.backends.base import ChunkRef, ChunkStore


class LocalDiskChunkStore(ChunkStore):
    def __init__(self, root_dir: str | Path, max_ready_chunks: int, low_watermark_chunks: int) -> None:
        self.root_dir = Path(root_dir)
        self.staging_dir = self.root_dir / "staging"
        self.ready_dir = self.root_dir / "ready"
        self.consumed_dir = self.root_dir / "consumed"

        self.max_ready_chunks = max(1, int(max_ready_chunks))
        self.low_watermark_chunks = max(0, int(low_watermark_chunks))
        if self.low_watermark_chunks > self.max_ready_chunks:
            self.low_watermark_chunks = self.max_ready_chunks

        for folder in (self.staging_dir, self.ready_dir, self.consumed_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def put_ready(self, chunk_id: str, local_file: str | Path, meta: dict[str, Any]) -> ChunkRef:
        source = Path(local_file)
        suffix = "".join(source.suffixes)
        filename = f"{chunk_id}{suffix}"

        staging_path = self.staging_dir / filename
        ready_path = self.ready_dir / filename

        shutil.copy2(source, staging_path)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.replace(ready_path)

        created_at = float(meta.get("created_at", time.time()))
        size_bytes = ready_path.stat().st_size
        return ChunkRef(
            chunk_id=chunk_id,
            uri=str(ready_path),
            size_bytes=size_bytes,
            created_at=created_at,
            backend_key=str(ready_path),
        )

    def list_ready(self, limit: int | None = None) -> list[ChunkRef]:
        files = [path for path in self.ready_dir.iterdir() if path.is_file()]
        files.sort(key=lambda item: (item.stat().st_mtime, item.name))
        if limit is not None:
            files = files[: max(0, int(limit))]

        refs: list[ChunkRef] = []
        for path in files:
            refs.append(
                ChunkRef(
                    chunk_id=_chunk_id_from_filename(path.name),
                    uri=str(path),
                    size_bytes=path.stat().st_size,
                    created_at=path.stat().st_mtime,
                    backend_key=str(path),
                )
            )
        return refs

    def fetch_to_local(self, chunk_ref: ChunkRef, target_path: str | Path) -> Path:
        source = Path(chunk_ref.backend_key)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, tmp_path)
        tmp_path.replace(target)
        return target

    def delete_ready(self, chunk_ref: ChunkRef) -> None:
        path = Path(chunk_ref.backend_key)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def count_ready(self) -> int:
        return len([path for path in self.ready_dir.iterdir() if path.is_file()])

    def capacity_state(self) -> dict[str, Any]:
        ready = self.count_ready()
        return {
            "backend": "local_disk",
            "ready_chunks": ready,
            "max_ready_chunks": self.max_ready_chunks,
            "low_watermark_chunks": self.low_watermark_chunks,
            "can_accept": ready < self.max_ready_chunks,
            "needs_fill": ready < self.low_watermark_chunks,
        }


def _chunk_id_from_filename(name: str) -> str:
    if name.endswith(".pt.gz"):
        return name[: -len(".pt.gz")]
    if name.endswith(".pt"):
        return name[: -len(".pt")]
    return Path(name).stem
