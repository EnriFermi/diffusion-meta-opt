from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.shared.streaming.backends.base import ChunkRef
from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.chunk_format import load_chunk
from dataset.shared.streaming.chunk_reader import ChunkReader
from dataset.shared.streaming.chunk_writer import ChunkWriter
from dataset.shared.types import SharedSample


def _make_sample(i: int) -> SharedSample:
    return SharedSample(
        model_name="clip_vit_b32",
        layer_name="vision_model.encoder.layers.0.mlp.fc1",
        weight=torch.zeros((3, 3), dtype=torch.float32),
        x=torch.ones((2, 3), dtype=torch.float32) * i,
        y=torch.ones((2, 3), dtype=torch.float32) * (i + 1),
        meta={"i": i},
    )


class TestStreamingChunkWriterReaderLocal(unittest.TestCase):
    def test_writer_reader_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalDiskChunkStore(root_dir=root / "store", max_ready_chunks=20, low_watermark_chunks=10)
            writer = ChunkWriter(
                store=store,
                spool_dir=root / "spool",
                chunk_size_samples=3,
                compression="none",
                spool_max_pending_chunks=10,
            )

            expected = 7
            for idx in range(expected):
                writer.append(_make_sample(idx))
            writer.flush(force_partial=True)

            self.assertGreater(store.count_ready(), 0)

            reader = ChunkReader(
                store=store,
                cache_dir=root / "consumer_cache",
                prefetch_max_chunks=2,
                delete_remote_after="consume",
                distributed_cfg={"enabled": False},
            )
            try:
                seen = []
                for _ in range(expected):
                    sample = reader.next_sample(block=True, timeout=1.0)
                    seen.append(int(sample.meta["i"]))

                self.assertEqual(sorted(seen), list(range(expected)))
                self.assertEqual(store.count_ready(), 0)
            finally:
                reader.close()
                writer.close()

    def test_chunk_contains_mixed_runs_and_datasets_with_window_striping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalDiskChunkStore(root_dir=root / "store", max_ready_chunks=20, low_watermark_chunks=10)
            writer = ChunkWriter(
                store=store,
                spool_dir=root / "spool",
                chunk_size_samples=3,
                compression="none",
                spool_max_pending_chunks=10,
            )

            writer.open_window(window_chunks=2, window_id=1)
            for idx in range(12):
                run_id = (idx // 2) % 2
                ds_name = "coco2017" if (idx // 2) % 2 == 0 else "scene_parse_150"
                sample = _make_sample(idx)
                sample.meta = {
                    "model_run_id": run_id,
                    "image_meta": [{"dataset_name": ds_name, "source_id": idx}],
                }
                writer.append(sample)
            writer.close_window(flush_partial=True)
            writer.close()

            refs = store.list_ready(limit=None)
            self.assertGreaterEqual(len(refs), 2)

            saw_mixed_run_chunk = False
            saw_mixed_dataset_chunk = False
            for ref in refs:
                samples, _ = load_chunk(Path(ref.backend_key))
                run_ids = {int(item.meta.get("model_run_id", -1)) for item in samples}
                ds_names = {
                    str(meta.get("dataset_name"))
                    for item in samples
                    for meta in item.meta.get("image_meta", [])
                    if meta.get("dataset_name") is not None
                }
                if len(run_ids) > 1:
                    saw_mixed_run_chunk = True
                if len(ds_names) > 1:
                    saw_mixed_dataset_chunk = True

            self.assertTrue(saw_mixed_run_chunk)
            self.assertTrue(saw_mixed_dataset_chunk)

    def test_reader_prefetch_scans_ready_only_once_per_fill_cycle(self) -> None:
        class _FakeStore:
            def __init__(self, refs: list[ChunkRef]) -> None:
                self.refs = refs
                self.list_ready_calls = 0

            def list_ready(self, limit: int | None = None) -> list[ChunkRef]:
                self.list_ready_calls += 1
                del limit
                return list(self.refs)

            def fetch_to_local(self, chunk_ref: ChunkRef, target_path: str | Path) -> Path:
                source = Path(chunk_ref.backend_key)
                target = Path(target_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                return target

            def delete_ready(self, chunk_ref: ChunkRef) -> None:
                del chunk_ref

            def count_ready(self) -> int:
                return len(self.refs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs: list[ChunkRef] = []
            for idx in range(3):
                chunk_path = root / f"chunk_{idx}.pt"
                save_samples = [_make_sample(idx)]
                from dataset.shared.streaming.chunk_format import save_chunk

                save_chunk(chunk_path, samples=save_samples, meta={"chunk_id": f"chunk_{idx}"}, compression="none")
                refs.append(
                    ChunkRef(
                        chunk_id=f"chunk_{idx}",
                        uri=str(chunk_path),
                        size_bytes=int(chunk_path.stat().st_size),
                        created_at=float(idx),
                        backend_key=str(chunk_path),
                    )
                )

            reader = ChunkReader(
                store=_FakeStore(refs),
                cache_dir=root / "cache",
                prefetch_max_chunks=3,
                delete_remote_after="consume",
                distributed_cfg={"enabled": False},
            )
            try:
                reader._prefetch_once()
                self.assertEqual(reader.store.list_ready_calls, 1)
                self.assertEqual(len(reader._pending_chunks), 3)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
