from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image

from dataset.data_raw.core.fs_utils import ensure_dir, read_json, slugify, stable_hash, utc_now_iso, write_json_atomic


class ChunkCache:
    """Disk-backed image chunk cache for a single dataset."""

    def __init__(self, dataset_root: str | Path, num_chunks_kept: int) -> None:
        self.dataset_root = ensure_dir(dataset_root)
        self.chunks_root = ensure_dir(self.dataset_root / "chunks")
        self.index_path = self.chunks_root / "index.json"
        self.num_chunks_kept = max(1, int(num_chunks_kept))
        self._active_chunk_id: str | None = None

        if not self.index_path.exists():
            self._write_index()

    def has_chunk(self, chunk_id: str) -> bool:
        return (self.chunks_root / chunk_id).is_dir()

    def chunk_path(self, chunk_id: str) -> Path:
        return self.chunks_root / chunk_id

    def lease_dir(self, chunk_id: str) -> Path:
        return self.chunk_path(chunk_id) / ".leases"

    def list_chunks(self) -> list[str]:
        chunks = [
            path.name
            for path in self._iter_chunk_dirs()
            if (path / "manifest.json").exists()
        ]
        chunks.sort()
        return chunks

    def cleanup_incomplete_chunks(self) -> list[str]:
        removed: list[str] = []
        for path in self._iter_chunk_dirs():
            if (path / "manifest.json").exists():
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        if removed:
            self._write_index()
        return removed

    def start_chunk(self, chunk_id: str | None = None) -> str:
        if chunk_id is None:
            chunk_id = f"chunk_{int(time.time() * 1000)}"
        ensure_dir(self.chunk_path(chunk_id))
        self._active_chunk_id = chunk_id
        self._write_index()
        return chunk_id

    def load_chunk(self, chunk_id: str) -> dict[str, Any]:
        manifest_path = self.chunk_path(chunk_id) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Chunk manifest does not exist: {manifest_path}")
        return read_json(manifest_path)

    def save_image(self, sample_id: str | int, image: Image.Image, chunk_id: str | None = None) -> Path:
        target_chunk_id = chunk_id or self._active_chunk_id
        if not target_chunk_id:
            raise ValueError("ChunkCache.save_image requires active chunk or explicit chunk_id")

        target_dir = ensure_dir(self.chunk_path(target_chunk_id))

        sample_str = str(sample_id)
        prefix = slugify(sample_str)
        suffix = stable_hash(sample_str)[:10]
        filename = f"{prefix}_{suffix}.jpg"
        image_path = target_dir / filename

        image = image.convert("RGB")
        image.save(image_path, format="JPEG", quality=95)
        return image_path

    def finalize_chunk(
        self,
        chunk_id: str,
        items: list[dict[str, Any]],
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chunk_id": chunk_id,
            "created_at": utc_now_iso(),
            "num_images": len(items),
            "items": items,
        }
        if extra_meta:
            payload.update(extra_meta)

        write_json_atomic(self.chunk_path(chunk_id) / "manifest.json", payload)
        self.evict_old_chunks()
        self._write_index()

    def remove_chunk(self, chunk_id: str) -> None:
        if self.has_active_leases(chunk_id):
            return
        chunk_dir = self.chunk_path(chunk_id)
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
        self._write_index()

    def evict_old_chunks(self) -> list[str]:
        evicted: list[str] = []
        chunks = self.list_chunks()
        while len(chunks) > self.num_chunks_kept:
            evictable = next((chunk_id for chunk_id in chunks if not self.has_active_leases(chunk_id)), None)
            if evictable is None:
                break
            chunks.remove(evictable)
            self.remove_chunk(evictable)
            evicted.append(evictable)
        if evicted:
            self._write_index()
        return evicted

    def count_images(self) -> int:
        total = 0
        for chunk_id in self.list_chunks():
            try:
                manifest = self.load_chunk(chunk_id)
            except Exception:
                continue
            total += int(manifest.get("num_images", 0))
        return total

    def _write_index(self) -> None:
        payload = {
            "updated_at": utc_now_iso(),
            "chunks": self.list_chunks(),
        }
        write_json_atomic(self.index_path, payload)

    def _iter_chunk_dirs(self) -> list[Path]:
        return [path for path in self.chunks_root.iterdir() if path.is_dir() and path.name.startswith("chunk_")]

    def acquire_lease(self, chunk_id: str, owner_id: str) -> None:
        lease_dir = ensure_dir(self.lease_dir(chunk_id))
        self._lease_path(chunk_id, owner_id).touch(exist_ok=True)

    def release_lease(self, chunk_id: str, owner_id: str) -> None:
        lease_path = self._lease_path(chunk_id, owner_id)
        try:
            lease_path.unlink(missing_ok=True)
        except Exception:
            pass
        lease_dir = self.lease_dir(chunk_id)
        try:
            if lease_dir.exists() and not any(lease_dir.iterdir()):
                lease_dir.rmdir()
        except Exception:
            pass

    def has_active_leases(self, chunk_id: str) -> bool:
        self._cleanup_stale_leases(chunk_id)
        lease_dir = self.lease_dir(chunk_id)
        if not lease_dir.exists():
            return False
        try:
            return any(lease_dir.iterdir())
        except Exception:
            return False

    def _lease_path(self, chunk_id: str, owner_id: str) -> Path:
        safe_owner = slugify(str(owner_id)) or "lease"
        return self.lease_dir(chunk_id) / f"{safe_owner}.lease"

    def _cleanup_stale_leases(self, chunk_id: str) -> None:
        lease_dir = self.lease_dir(chunk_id)
        if not lease_dir.exists():
            return
        for path in lease_dir.iterdir():
            if not path.is_file():
                continue
            owner_id = path.stem
            pid = _parse_owner_pid(owner_id)
            if pid is None or _pid_is_alive(pid):
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                continue
        try:
            if not any(lease_dir.iterdir()):
                lease_dir.rmdir()
        except Exception:
            pass


def _parse_owner_pid(owner_id: str) -> int | None:
    token = str(owner_id).strip()
    if not token.startswith("pid"):
        return None
    pid_text = token[3:].split("_", 1)[0]
    try:
        return int(pid_text)
    except Exception:
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True
    return True
