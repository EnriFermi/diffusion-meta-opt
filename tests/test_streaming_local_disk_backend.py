from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset.shared.streaming.backends.local_disk import LocalDiskChunkStore


class TestStreamingLocalDiskBackend(unittest.TestCase):
    def test_put_list_fetch_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "store"
            source = Path(tmp) / "source_chunk.pt"
            source.write_bytes(b"payload")

            store = LocalDiskChunkStore(root_dir=root, max_ready_chunks=8, low_watermark_chunks=4)
            ref = store.put_ready("chunk_001", local_file=source, meta={"created_at": 123.0})

            refs = store.list_ready(limit=None)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].chunk_id, "chunk_001")
            self.assertEqual(store.count_ready(), 1)
            self.assertEqual(ref.chunk_id, "chunk_001")

            fetched = Path(tmp) / "fetched.pt"
            store.fetch_to_local(refs[0], fetched)
            self.assertEqual(fetched.read_bytes(), b"payload")

            store.delete_ready(refs[0])
            self.assertEqual(store.count_ready(), 0)
            self.assertEqual(len(store.list_ready(limit=None)), 0)

            staging_files = [path for path in (root / "staging").iterdir() if path.is_file()]
            self.assertEqual(staging_files, [])

    def test_capacity_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "store"
            store = LocalDiskChunkStore(root_dir=root, max_ready_chunks=2, low_watermark_chunks=1)

            source = Path(tmp) / "chunk.pt"
            source.write_bytes(b"x")

            store.put_ready("chunk_a", local_file=source, meta={})
            state = store.capacity_state()
            self.assertTrue(state["can_accept"])
            self.assertFalse(state["needs_fill"])

            store.put_ready("chunk_b", local_file=source, meta={})
            state = store.capacity_state()
            self.assertFalse(state["can_accept"])


if __name__ == "__main__":
    unittest.main()
