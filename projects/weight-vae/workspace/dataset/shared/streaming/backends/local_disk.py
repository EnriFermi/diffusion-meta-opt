from __future__ import annotations

import json
import os
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
        staging_tmp_path = staging_path.with_name(f"{staging_path.name}.tmp")

        try:
            staging_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            os.link(source, staging_tmp_path)
        except Exception:
            shutil.copy2(source, staging_tmp_path)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        staging_tmp_path.replace(ready_path)

        created_at = float(meta.get("created_at", time.time()))
        size_bytes = ready_path.stat().st_size
        self._write_meta_sidecar(
            chunk_id=chunk_id,
            ready_path=ready_path,
            meta=meta,
            created_at=created_at,
            size_bytes=size_bytes,
        )
        return ChunkRef(
            chunk_id=chunk_id,
            uri=str(ready_path),
            size_bytes=size_bytes,
            created_at=created_at,
            backend_key=str(ready_path),
        )

    def list_ready(self, limit: int | None = None) -> list[ChunkRef]:
        files_with_meta: list[tuple[float, str, Path, int]] = []
        for entry in os.scandir(self.ready_dir):
            name = entry.name
            if not _is_ready_chunk_entry(entry):
                continue
            try:
                stat = entry.stat()
            except FileNotFoundError:
                continue
            files_with_meta.append((float(stat.st_mtime), name, Path(entry.path), int(stat.st_size)))

        files_with_meta.sort(key=lambda item: (item[0], item[1]))
        if limit is not None:
            files_with_meta = files_with_meta[: max(0, int(limit))]

        refs: list[ChunkRef] = []
        for mtime, _, path, size_bytes in files_with_meta:
            refs.append(
                ChunkRef(
                    chunk_id=_chunk_id_from_filename(path.name),
                    uri=str(path),
                    size_bytes=size_bytes,
                    created_at=mtime,
                    backend_key=str(path),
                )
            )
        return refs

    def fetch_to_local(self, chunk_ref: ChunkRef, target_path: str | Path) -> Path:
        source = Path(chunk_ref.backend_key)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            os.link(source, tmp_path)
        except Exception:
            shutil.copy2(source, tmp_path)
        tmp_path.replace(target)
        return target

    def delete_ready(self, chunk_ref: ChunkRef) -> None:
        path = Path(chunk_ref.backend_key)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self._meta_sidecar_path(chunk_ref.chunk_id).unlink(missing_ok=True)
        except Exception:
            pass

    def count_ready(self) -> int:
        count = 0
        for entry in os.scandir(self.ready_dir):
            if _is_ready_chunk_entry(entry):
                count += 1
        return count

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

    def debug_snapshot(self, limit: int = 10) -> dict[str, Any]:
        limit_int = max(1, int(limit))
        staging_files = sorted(
            [path.name for path in self.staging_dir.iterdir() if path.is_file()],
        )
        ready_files = sorted(
            [path.name for path in self.ready_dir.iterdir() if _is_ready_chunk_file(path)],
        )
        consumed_files = sorted(
            [path.name for path in self.consumed_dir.iterdir() if path.is_file()],
        )
        return {
            "staging_count": len(staging_files),
            "ready_count": len(ready_files),
            "consumed_count": len(consumed_files),
            "staging_head": staging_files[:limit_int],
            "ready_head": ready_files[:limit_int],
            "consumed_head": consumed_files[:limit_int],
        }

    def read_ready_chunk_meta(self, chunk_id: str) -> dict[str, Any] | None:
        path = self._meta_sidecar_path(chunk_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _meta_sidecar_path(self, chunk_id: str) -> Path:
        safe_chunk_id = str(chunk_id).replace("/", "_")
        return self.ready_dir / f"{safe_chunk_id}.meta.json"

    def _write_meta_sidecar(
        self,
        chunk_id: str,
        ready_path: Path,
        meta: dict[str, Any],
        created_at: float,
        size_bytes: int,
    ) -> None:
        sidecar_path = self._meta_sidecar_path(chunk_id)
        payload = {
            "chunk_id": str(chunk_id),
            "created_at": float(created_at),
            "size_bytes": int(size_bytes),
            "ready_path": str(ready_path),
            "meta": meta,
        }
        tmp_path = sidecar_path.with_name(sidecar_path.name + ".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            tmp_path.replace(sidecar_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _chunk_id_from_filename(name: str) -> str:
    if name.endswith(".pt.gz"):
        return name[: -len(".pt.gz")]
    if name.endswith(".pt"):
        return name[: -len(".pt")]
    return Path(name).stem


def _is_ready_chunk_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
    except FileNotFoundError:
        return False

    name = path.name
    if name.endswith(".tmp"):
        return False
    if name.endswith(".meta.json"):
        return False
    return name.endswith(".pt") or name.endswith(".pt.gz")


def _is_ready_chunk_entry(entry: os.DirEntry[str]) -> bool:
    try:
        if not entry.is_file():
            return False
    except FileNotFoundError:
        return False

    name = entry.name
    if name.endswith(".tmp"):
        return False
    if name.endswith(".meta.json"):
        return False
    return name.endswith(".pt") or name.endswith(".pt.gz")
