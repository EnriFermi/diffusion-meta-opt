from __future__ import annotations

import unittest

import torch

from dataset.models.types import LayerIORecord
from dataset.shared.atomizer import atomize
from dataset.shared.types import MixedImageMeta


class TestAtomizer(unittest.TestCase):
    def test_random_slice_atomization(self) -> None:
        record = LayerIORecord(
            model_name="m",
            layer_name="l",
            weight=torch.ones(2, 3),
            inputs=torch.randn(5, 3),
            outputs=torch.randn(5, 2),
            meta={"input_shape_list": [(5, 3)], "num_calls": 1},
        )

        image_meta = [MixedImageMeta(dataset_name="ds", source_id=i) for i in range(5)]
        items = list(
            atomize(
                record,
                {"xy_samples_random_slice": 2},
                image_meta,
                model_run_id=7,
            )
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].x.shape[0], 2)
        self.assertEqual(items[0].y.shape[0], 2)
        self.assertEqual(items[0].meta["model_run_id"], 7)
        self.assertEqual(items[0].meta["xy_sampling_mode"], "random_slice")
        for item in items:
            self.assertEqual(tuple(item.weight.shape), tuple(record.weight.shape))
            self.assertTrue(torch.equal(item.weight, record.weight))

    def test_full_when_rows_less_or_equal_than_slice_size(self) -> None:
        record = LayerIORecord(
            model_name="m",
            layer_name="l",
            weight=torch.ones(2, 3),
            inputs=torch.randn(3, 3),
            outputs=torch.randn(3, 2),
            meta={"input_shape_list": [(3, 3)], "num_calls": 1},
        )

        image_meta = [MixedImageMeta(dataset_name="ds", source_id=i) for i in range(3)]
        items = list(atomize(record, {"xy_samples_random_slice": 8}, image_meta, model_run_id=5))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].x.shape[0], 3)
        self.assertEqual(items[0].y.shape[0], 3)
        self.assertEqual(items[0].meta["xy_sampling_mode"], "full")


if __name__ == "__main__":
    unittest.main()
