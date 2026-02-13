from __future__ import annotations

import unittest

from dataset.shared.cache import SharedSampleCache


class TestSharedCache(unittest.TestCase):
    def test_hysteresis_and_size(self) -> None:
        cache = SharedSampleCache(max_items=4, fill_target=3, low_watermark=2)

        self.assertTrue(cache.needs_fill())
        self.assertTrue(cache.is_low())

        cache.put("a")
        cache.put("b")
        self.assertEqual(cache.size(), 2)
        self.assertTrue(cache.needs_fill())
        self.assertFalse(cache.is_low())

        cache.put("c")
        self.assertEqual(cache.size(), 3)
        self.assertFalse(cache.needs_fill())

        _ = cache.get()
        self.assertEqual(cache.size(), 2)


if __name__ == "__main__":
    unittest.main()
