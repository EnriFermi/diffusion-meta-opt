from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
from dataset.shared.streaming.chunk_format import load_chunk
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


class TestStreamingChunkWriterRoundRobin(unittest.TestCase):
    def test_window_round_robin_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalDiskChunkStore(root_dir=root / "store", max_ready_chunks=100, low_watermark_chunks=50)
            writer = ChunkWriter(
                store=store,
                spool_dir=root / "spool",
                chunk_size_samples=2,
                compression="none",
                spool_max_pending_chunks=100,
            )

            writer.open_window(window_chunks=3, window_id=12)
            for i in range(10):
                writer.append(_sample(i))
            writer.close_window(flush_partial=True)
            writer.close()

            refs = store.list_ready(limit=None)
            self.assertGreaterEqual(len(refs), 3)

            seen_ids: set[int] = set()
            for ref in refs:
                samples, info = load_chunk(Path(ref.backend_key))
                meta = dict(info.get("meta", {}))
                slot = int(meta["window_slot"])
                self.assertEqual(int(meta["window_id"]), 12)
                self.assertEqual(int(meta["window_chunks"]), 3)

                chunk_indices = [int(item.meta["i"]) for item in samples]
                for idx in chunk_indices:
                    self.assertEqual(idx % 3, slot)
                    seen_ids.add(idx)

            self.assertEqual(seen_ids, set(range(10)))


if __name__ == "__main__":
    unittest.main()
