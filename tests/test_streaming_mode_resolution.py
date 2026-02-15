from __future__ import annotations

import os
import unittest

from dataset.shared.streaming.config import normalize_streaming_mode, resolve_distributed_settings
from dataset.shared.streaming.factory import resolve_streaming_cfg


class TestStreamingModeResolution(unittest.TestCase):
    def test_alias_in_memory_maps_to_none(self) -> None:
        self.assertEqual(normalize_streaming_mode("in_memory"), "none")
        self.assertEqual(normalize_streaming_mode("none"), "none")

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_streaming_mode("bad_mode")

    def test_resolve_streaming_cfg_defaults(self) -> None:
        cfg = resolve_streaming_cfg({})
        self.assertEqual(cfg["mode"], "none")
        self.assertEqual(cfg["chunk_size_samples"], 256)
        self.assertIn("refill_after_consumed_chunks", cfg["producer"])
        self.assertEqual(cfg["consumer"]["randomize_within_chunk"], True)

    def test_resolve_streaming_cfg_alias(self) -> None:
        cfg = resolve_streaming_cfg({"mode": "in_memory"})
        self.assertEqual(cfg["mode"], "none")

    def test_distributed_env_resolution(self) -> None:
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "2"
        try:
            settings = resolve_distributed_settings(
                {
                    "enabled": True,
                    "rank_env": "RANK",
                    "world_size_env": "WORLD_SIZE",
                    "shard_by": "chunk",
                }
            )
        finally:
            os.environ.pop("RANK", None)
            os.environ.pop("WORLD_SIZE", None)

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.rank, 1)
        self.assertEqual(settings.world_size, 2)
        self.assertEqual(settings.shard_by, "chunk")


if __name__ == "__main__":
    unittest.main()
