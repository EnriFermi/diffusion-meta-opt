from __future__ import annotations

import unittest

import torch

from dataset.models.types import LayerIORecord
from dataset.shared.atomizer import atomize
from dataset.shared.types import MixedImageMeta


class TestAtomizerRandomSlice(unittest.TestCase):
    def test_selected_pairs_are_linked_and_unique(self) -> None:
        # Build deterministic linked x/y rows so we can verify index linkage.
        x = torch.arange(0, 60, dtype=torch.float32).reshape(20, 3)
        y = (torch.arange(0, 40, dtype=torch.float32).reshape(20, 2) + 1000.0)
        record = LayerIORecord(
            model_name="m",
            layer_name="l",
            weight=torch.randn(4, 3),
            inputs=x,
            outputs=y,
            meta={"input_shape_list": [(20, 3)], "num_calls": 1},
        )

        image_meta = [MixedImageMeta(dataset_name="ds", source_id=i) for i in range(20)]
        items = list(atomize(record, {"xy_samples_random_slice": 7}, image_meta, model_run_id=1))
        self.assertEqual(len(items), 1)
        sample = items[0]

        self.assertEqual(sample.x.shape[0], 7)
        self.assertEqual(sample.y.shape[0], 7)

        selected_idx = sample.meta["selected_row_indices_preview"]
        self.assertEqual(len(selected_idx), len(set(selected_idx)))
        self.assertTrue(all(isinstance(v, int) for v in selected_idx))

        # linkage check: values in sampled rows must correspond to the same indices in source.
        for row_i, src_idx in enumerate(selected_idx):
            self.assertTrue(torch.equal(sample.x[row_i], x[src_idx]))
            self.assertTrue(torch.equal(sample.y[row_i], y[src_idx]))

    def test_full_batch_returned_when_n_le_k(self) -> None:
        x = torch.randn(5, 3)
        y = torch.randn(5, 2)
        record = LayerIORecord(
            model_name="m",
            layer_name="l",
            weight=torch.randn(2, 3),
            inputs=x,
            outputs=y,
            meta={},
        )
        image_meta = [MixedImageMeta(dataset_name="ds", source_id=i) for i in range(5)]
        items = list(atomize(record, {"xy_samples_random_slice": 10}, image_meta, model_run_id=2))
        self.assertEqual(len(items), 1)
        sample = items[0]
        self.assertEqual(sample.x.shape[0], 5)
        self.assertEqual(sample.y.shape[0], 5)
        self.assertEqual(sample.meta["xy_sampling_mode"], "full")


if __name__ == "__main__":
    unittest.main()

