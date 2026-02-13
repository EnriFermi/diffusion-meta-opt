from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore
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
                local_max_chunks=10,
            )

            expected = 7
            for idx in range(expected):
                writer.append(_make_sample(idx))
            writer.flush(force_partial=True)

            self.assertGreater(store.count_ready(), 0)

            reader = ChunkReader(
                store=store,
                local_cache_dir=root / "consumer_cache",
                local_max_chunks=2,
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


if __name__ == "__main__":
    unittest.main()
