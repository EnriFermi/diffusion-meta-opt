from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.shared.streaming.chunk_format import load_chunk, save_chunk
from dataset.shared.types import SharedSample


def _sample(i: int) -> SharedSample:
    return SharedSample(
        model_name="clip_vit_b32",
        layer_name=f"layer_{i}",
        weight=torch.ones((2, 2), dtype=torch.float32) * i,
        x=torch.randn(4, 2, dtype=torch.float32),
        y=torch.randn(4, 2, dtype=torch.float32),
        meta={"sample_index": i},
    )


class TestStreamingChunkFormat(unittest.TestCase):
    def test_roundtrip_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk_a.pt"
            original = [_sample(0), _sample(1)]
            save_chunk(path, samples=original, meta={"chunk_id": "chunk_a"}, compression="none")

            loaded, meta = load_chunk(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(meta["chunk_id"], "chunk_a")
            self.assertTrue(torch.equal(loaded[0].weight, original[0].weight))
            self.assertEqual(loaded[1].meta["sample_index"], 1)

    def test_roundtrip_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk_b.pt.gz"
            original = [_sample(2), _sample(3)]
            save_chunk(path, samples=original, meta={"chunk_id": "chunk_b"}, compression="gzip")

            loaded, meta = load_chunk(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(meta["chunk_id"], "chunk_b")
            self.assertEqual(loaded[0].layer_name, "layer_2")
            self.assertEqual(loaded[1].layer_name, "layer_3")


if __name__ == "__main__":
    unittest.main()
