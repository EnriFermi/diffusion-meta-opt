from __future__ import annotations

import logging
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset.shared.streaming.backends.base import ChunkStore
from dataset.shared.streaming.chunk_format import chunk_suffix, save_chunk
from dataset.shared.types import SharedSample


@dataclass(slots=True)
class _SpoolItem:
    chunk_id: str
    local_path: Path
    meta: dict[str, Any]


class ChunkWriter:
    def __init__(
        self,
        store: ChunkStore,
        spool_dir: str | Path,
        chunk_size_samples: int,
        compression: str,
        spool_max_pending_chunks: int,
    ) -> None:
        self.store = store
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size_samples = max(1, int(chunk_size_samples))
        self.compression = str(compression or "none").lower()
        self.spool_max_pending_chunks = max(1, int(spool_max_pending_chunks))

        self.logger = logging.getLogger(self.__class__.__name__)

        self._window_open = False
        self._window_chunks = 1
        self._window_id = 0
        self._window_cursor = 0
        self._slot_buffers: dict[int, list[SharedSample]] = {0: []}

        self._sequence = 0
        self._num_chunks_written = 0
        self._num_chunks_published = 0
        self._spool_items: deque[_SpoolItem] = deque()

    def open_window(self, window_chunks: int, window_id: int) -> None:
        normalized_chunks = max(1, int(window_chunks))
        normalized_window_id = max(0, int(window_id))

        if self._window_open and normalized_chunks == self._window_chunks and normalized_window_id == self._window_id:
            return

        if normalized_chunks != self._window_chunks and any(self._slot_buffers.get(slot) for slot in self._slot_buffers):
            # Window size changed while buffers are non-empty: flush remainder first to avoid remapping ambiguity.
            self.flush(force_partial=True)

        self._window_chunks = normalized_chunks
        self._window_id = normalized_window_id
        self._window_cursor = 0
        self._window_open = True
        self._ensure_slot_buffers()

    def close_window(self, flush_partial: bool = False) -> None:
        self.flush(force_partial=bool(flush_partial))
        self._window_open = False

    def append(self, sample: SharedSample) -> None:
        self._drain_spool()

        self._ensure_window_open()
        self._ensure_slot_buffers()

        slot = self._window_cursor % self._window_chunks
        self._window_cursor += 1

        slot_buffer = self._slot_buffers[slot]
        slot_buffer.append(sample)
        if len(slot_buffer) >= self.chunk_size_samples:
            self.flush(force_partial=False)

    def flush(self, force_partial: bool = False) -> int:
        emitted_chunks = 0
        self._drain_spool()

        self._ensure_slot_buffers()

        while True:
            emitted_in_pass = 0
            for slot in range(self._window_chunks):
                slot_buffer = self._slot_buffers[slot]
                while len(slot_buffer) >= self.chunk_size_samples:
                    batch = slot_buffer[: self.chunk_size_samples]
                    if not self._enqueue_chunk(batch, slot=slot):
                        return emitted_chunks
                    del slot_buffer[: self.chunk_size_samples]
                    emitted_chunks += 1
                    emitted_in_pass += 1
                    self._drain_spool()
            if emitted_in_pass == 0:
                break

        if force_partial:
            for slot in range(self._window_chunks):
                slot_buffer = self._slot_buffers[slot]
                if not slot_buffer:
                    continue
                if not self._enqueue_chunk(slot_buffer[:], slot=slot):
                    return emitted_chunks
                slot_buffer.clear()
                emitted_chunks += 1
                self._drain_spool()

        return emitted_chunks

    def can_accept_more(self) -> bool:
        return self._spool_size() < self.spool_max_pending_chunks

    def pending_samples(self) -> int:
        return sum(len(items) for items in self._slot_buffers.values())

    def close(self) -> None:
        self.flush(force_partial=True)
        self._drain_spool()

    def stats(self) -> dict[str, Any]:
        return {
            "pending_samples": self.pending_samples(),
            "num_chunks_written": self._num_chunks_written,
            "num_chunks_published": self._num_chunks_published,
            "chunk_size_samples": self.chunk_size_samples,
            "compression": self.compression,
            "spool_dir": str(self.spool_dir),
            "spool_pending_chunks": self._spool_size(),
            "spool_max_pending_chunks": self.spool_max_pending_chunks,
            "window_open": self._window_open,
            "window_chunks": self._window_chunks,
            "window_id": self._window_id,
            "window_cursor": self._window_cursor,
        }

    def _enqueue_chunk(self, samples: list[SharedSample], slot: int) -> bool:
        if not samples:
            return True

        if not self.can_accept_more():
            return False

        chunk_id = _new_chunk_id(window_id=self._window_id, slot=slot, sequence=self._sequence)
        self._sequence += 1

        chunk_summary = _build_chunk_sample_summary(samples)
        local_path = self.spool_dir / f"{chunk_id}{chunk_suffix(self.compression)}"
        meta = {
            "chunk_id": chunk_id,
            "created_at": time.time(),
            "num_samples": len(samples),
            "window_id": self._window_id,
            "window_slot": int(slot),
            "window_chunks": int(self._window_chunks),
            "sample_summary": chunk_summary,
        }

        save_chunk(local_path, samples=samples, meta=meta, compression=self.compression)
        self._spool_items.append(_SpoolItem(chunk_id=chunk_id, local_path=local_path, meta=meta))
        self._num_chunks_written += 1
        return True

    def _drain_spool(self) -> None:
        while self._spool_items:
            state = self.store.capacity_state()
            if not bool(state.get("can_accept", True)):
                break

            item = self._spool_items[0]
            try:
                self.store.put_ready(chunk_id=item.chunk_id, local_file=item.local_path, meta=item.meta)
            except Exception as exc:
                self.logger.warning("Failed to publish chunk %s: %s", item.chunk_id, exc)
                break

            try:
                item.local_path.unlink(missing_ok=True)
            except Exception:
                pass

            self._spool_items.popleft()
            self._num_chunks_published += 1

    def _spool_size(self) -> int:
        return len(self._spool_items)


    def _ensure_slot_buffers(self) -> None:
        for slot in range(self._window_chunks):
            self._slot_buffers.setdefault(slot, [])
        stale = [slot for slot in self._slot_buffers if slot >= self._window_chunks]
        for slot in stale:
            if self._slot_buffers[slot]:
                raise RuntimeError(f"Found stale non-empty slot buffer: slot={slot}, window_chunks={self._window_chunks}")
            del self._slot_buffers[slot]

    def _ensure_window_open(self) -> None:
        if self._window_open:
            return
        self.open_window(window_chunks=self._window_chunks, window_id=self._window_id)


def _new_chunk_id(window_id: int, slot: int, sequence: int) -> str:
    return f"final_chunk_w{int(window_id):05d}_s{int(slot):03d}_{int(time.time() * 1000)}_{sequence:08d}"


def _build_chunk_sample_summary(samples: list[SharedSample], preview_items: int = 4) -> dict[str, Any]:
    model_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    model_run_ids: set[int] = set()
    sample_preview: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        model_counts[str(sample.model_name)] += 1
        layer_counts[str(sample.layer_name)] += 1

        meta = sample.meta if isinstance(sample.meta, dict) else {}
        run_id = meta.get("model_run_id")
        if run_id is not None:
            try:
                model_run_ids.add(int(run_id))
            except Exception:
                pass

        image_meta = meta.get("image_meta", [])
        datasets: set[str] = set()
        if isinstance(image_meta, list):
            for item in image_meta:
                if not isinstance(item, dict):
                    continue
                ds_name = item.get("dataset_name")
                if ds_name is not None:
                    datasets.add(str(ds_name))
        for ds_name in datasets:
            dataset_counts[ds_name] += 1

        if idx < int(preview_items):
            sample_preview.append(
                {
                    "sample_idx": int(idx),
                    "model_name": str(sample.model_name),
                    "layer_name": str(sample.layer_name),
                    "datasets": sorted(datasets),
                    "model_run_id": int(run_id) if run_id is not None else None,
                    "xy_sampling_mode": meta.get("xy_sampling_mode"),
                    "selected_row_count": int(meta.get("selected_row_count", 0))
                    if meta.get("selected_row_count") is not None
                    else None,
                    "selected_row_indices_preview": list(meta.get("selected_row_indices_preview", []))[:8]
                    if isinstance(meta.get("selected_row_indices_preview"), list)
                    else [],
                }
            )

    return {
        "num_samples": int(len(samples)),
        "unique_models": int(len(model_counts)),
        "unique_layers": int(len(layer_counts)),
        "unique_datasets": int(len(dataset_counts)),
        "unique_model_run_ids": int(len(model_run_ids)),
        "top_models": [[name, int(count)] for name, count in model_counts.most_common(5)],
        "top_layers": [[name, int(count)] for name, count in layer_counts.most_common(5)],
        "top_datasets": [[name, int(count)] for name, count in dataset_counts.most_common(5)],
        "sample_preview": sample_preview,
    }
