from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.chunk_reader import ChunkReader
from dataset.shared.streaming.chunk_writer import ChunkWriter
from dataset.shared.types import SharedSample


def _sample(i: int) -> SharedSample:
    return SharedSample(
        model_name="clip_vit_b32",
        layer_name="vision_model.encoder.layers.0.mlp.fc1",
        weight=torch.zeros((2, 2), dtype=torch.float32),
        x=torch.ones((1, 2), dtype=torch.float32) * i,
        y=torch.ones((1, 2), dtype=torch.float32) * (i + 1),
        meta={"i": i},
    )


class TestStreamingChunkReaderShuffle(unittest.TestCase):
    def test_shuffle_within_chunk_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalDiskChunkStore(root_dir=root / "store", max_ready_chunks=20, low_watermark_chunks=10)
            writer = ChunkWriter(
                store=store,
                spool_dir=root / "spool",
                chunk_size_samples=10,
                compression="none",
                spool_max_pending_chunks=20,
            )
            writer.open_window(window_chunks=1, window_id=0)
            for i in range(10):
                writer.append(_sample(i))
            writer.close_window(flush_partial=True)

            reader = ChunkReader(
                store=store,
                cache_dir=root / "consumer_cache",
                prefetch_max_chunks=1,
                delete_remote_after="consume",
                distributed_cfg={"enabled": False},
                randomize_within_chunk=True,
                random_seed=123,
            )
            try:
                order = [int(reader.next_sample(block=True, timeout=1.0).meta["i"]) for _ in range(10)]
            finally:
                reader.close()
                writer.close()

            self.assertEqual(set(order), set(range(10)))
            self.assertNotEqual(order, list(range(10)))

    def test_shuffle_within_chunk_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalDiskChunkStore(root_dir=root / "store", max_ready_chunks=20, low_watermark_chunks=10)
            writer = ChunkWriter(
                store=store,
                spool_dir=root / "spool",
                chunk_size_samples=10,
                compression="none",
                spool_max_pending_chunks=20,
            )
            writer.open_window(window_chunks=1, window_id=0)
            for i in range(10):
                writer.append(_sample(i))
            writer.close_window(flush_partial=True)

            reader = ChunkReader(
                store=store,
                cache_dir=root / "consumer_cache",
                prefetch_max_chunks=1,
                delete_remote_after="consume",
                distributed_cfg={"enabled": False},
                randomize_within_chunk=False,
                random_seed=123,
            )
            try:
                order = [int(reader.next_sample(block=True, timeout=1.0).meta["i"]) for _ in range(10)]
            finally:
                reader.close()
                writer.close()

            self.assertEqual(order, list(range(10)))

    def test_randomize_chunk_order_changes_first_chunk_across_seeds(self) -> None:
        first_chunk_prefixes: set[int] = set()

        for seed in range(6):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = LocalDiskChunkStore(root_dir=root / "store", max_ready_chunks=20, low_watermark_chunks=10)
                writer = ChunkWriter(
                    store=store,
                    spool_dir=root / "spool",
                    chunk_size_samples=5,
                    compression="none",
                    spool_max_pending_chunks=20,
                )
                writer.open_window(window_chunks=1, window_id=0)
                for i in range(10):
                    writer.append(_sample(i))
                writer.close_window(flush_partial=True)

                reader = ChunkReader(
                    store=store,
                    cache_dir=root / "consumer_cache",
                    prefetch_max_chunks=2,
                    delete_remote_after="consume",
                    distributed_cfg={"enabled": False},
                    randomize_chunk_order=True,
                    randomize_within_chunk=False,
                    random_seed=seed,
                )
                try:
                    first = int(reader.next_sample(block=True, timeout=1.0).meta["i"])
                finally:
                    reader.close()
                    writer.close()

                first_chunk_prefixes.add(first // 5)

        self.assertGreaterEqual(len(first_chunk_prefixes), 2)


if __name__ == "__main__":
    unittest.main()
