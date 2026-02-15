from __future__ import annotations

import random
import unittest

from dataset.shared.raw_dataset_pool import _normalize_max_dataset_fraction_per_batch


class TestRawDatasetPoolMixing(unittest.TestCase):
    def test_fraction_fallback_when_cap_times_n_below_one(self) -> None:
        # Requested cap is impossible for 2 datasets: 0.4 * 2 < 1.
        effective = _normalize_max_dataset_fraction_per_batch(requested=0.4, num_datasets=2)
        self.assertGreaterEqual(effective, 0.5)

    def test_fraction_kept_when_already_feasible(self) -> None:
        effective = _normalize_max_dataset_fraction_per_batch(requested=0.7, num_datasets=2)
        self.assertAlmostEqual(effective, 0.7, places=6)

    def test_fraction_clipped_for_single_dataset(self) -> None:
        effective = _normalize_max_dataset_fraction_per_batch(requested=0.2, num_datasets=1)
        self.assertEqual(effective, 1.0)

    def test_fraction_clipped_to_one(self) -> None:
        effective = _normalize_max_dataset_fraction_per_batch(requested=2.0, num_datasets=3)
        self.assertEqual(effective, 1.0)

    def test_random_seed_stability_not_required_for_normalizer(self) -> None:
        random.seed(123)
        a = _normalize_max_dataset_fraction_per_batch(requested=0.2, num_datasets=4)
        random.seed(999)
        b = _normalize_max_dataset_fraction_per_batch(requested=0.2, num_datasets=4)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

