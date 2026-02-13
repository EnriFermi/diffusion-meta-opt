from __future__ import annotations

import unittest

from dataset.shared.model_scheduler import ModelScheduler


class TestModelScheduler(unittest.TestCase):
    def test_shuffled_cycle_burst(self) -> None:
        scheduler = ModelScheduler(
            model_names=["m1", "m2"],
            model_weights={"m1": 1.0, "m2": 1.0},
            policy="shuffled_cycle",
            burst_jobs=2,
            seed=123,
        )

        first = scheduler.next_model()
        second = scheduler.next_model()
        self.assertEqual(first, second)

    def test_weighted_avoids_same_model_after_burst_boundary(self) -> None:
        scheduler = ModelScheduler(
            model_names=["m1", "m2"],
            model_weights={"m1": 1.0, "m2": 1.0},
            policy="weighted",
            burst_jobs=1,
            seed=42,
        )

        a = scheduler.next_model()
        b = scheduler.next_model()
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
