from __future__ import annotations

import unittest

from dataset.shared.streaming.chunk_reader import assign_chunk_rank


class TestStreamingSharding(unittest.TestCase):
    def test_assignment_is_deterministic(self) -> None:
        value_a = assign_chunk_rank("final_chunk_abc", world_size=4)
        value_b = assign_chunk_rank("final_chunk_abc", world_size=4)
        self.assertEqual(value_a, value_b)
        self.assertTrue(0 <= value_a < 4)

    def test_assignment_covers_multiple_ranks(self) -> None:
        world_size = 3
        assignments = {assign_chunk_rank(f"chunk_{idx}", world_size=world_size) for idx in range(50)}
        self.assertTrue(assignments.issubset({0, 1, 2}))
        self.assertGreaterEqual(len(assignments), 2)

    def test_world_size_one(self) -> None:
        self.assertEqual(assign_chunk_rank("any_chunk", world_size=1), 0)


if __name__ == "__main__":
    unittest.main()
