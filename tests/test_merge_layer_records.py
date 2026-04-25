from __future__ import annotations

import unittest

import torch

from dataset.models.model_runner import merge_layer_records
from dataset.models.types import LayerIORecord


class TestMergeLayerRecords(unittest.TestCase):
    def test_merge_layer_records(self) -> None:
        records = [
            LayerIORecord(
                model_name="m1",
                layer_name="l1",
                weight=torch.ones(2, 3),
                inputs=torch.ones(4, 3),
                outputs=torch.ones(4, 2),
                meta={"num_calls": 1, "input_shape_list": [(2, 2, 3)], "output_shape_list": [(2, 2, 2)]},
            ),
            LayerIORecord(
                model_name="m1",
                layer_name="l1",
                weight=torch.ones(2, 3),
                inputs=torch.ones(3, 3),
                outputs=torch.ones(3, 2),
                meta={"num_calls": 1, "input_shape_list": [(3, 3)], "output_shape_list": [(3, 2)]},
            ),
        ]

        merged = merge_layer_records(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(tuple(merged[0].inputs.shape), (7, 3))
        self.assertEqual(tuple(merged[0].outputs.shape), (7, 2))
        self.assertEqual(merged[0].meta["num_calls"], 2)

    def test_merge_layer_records_handles_zero_row_microbatch(self) -> None:
        records = [
            LayerIORecord(
                model_name="m1",
                layer_name="l1",
                weight=torch.ones(2, 3),
                inputs=torch.empty(0, 3),
                outputs=torch.empty(0, 2),
                meta={"num_calls": 0, "input_shape_list": [], "output_shape_list": []},
            ),
            LayerIORecord(
                model_name="m1",
                layer_name="l1",
                weight=torch.ones(2, 3),
                inputs=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                outputs=torch.tensor([[4.0, 5.0]], dtype=torch.float32),
                meta={"num_calls": 1, "input_shape_list": [(1, 3)], "output_shape_list": [(1, 2)]},
            ),
        ]

        merged = merge_layer_records(records)

        self.assertEqual(len(merged), 1)
        self.assertEqual(tuple(merged[0].inputs.shape), (1, 3))
        self.assertEqual(tuple(merged[0].outputs.shape), (1, 2))
        self.assertTrue(torch.equal(merged[0].inputs, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
