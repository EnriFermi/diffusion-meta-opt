from __future__ import annotations

import unittest

import torch

from dataset.models.types import LayerIORecord
from dataset.shared.atomizer import atomize
from dataset.shared.types import MixedImageMeta


class TestAtomizer(unittest.TestCase):
    def test_chunk_atomization(self) -> None:
        record = LayerIORecord(
            model_name="m",
            layer_name="l",
            weight=torch.ones(2, 3),
            inputs=torch.randn(5, 3),
            outputs=torch.randn(5, 2),
            meta={"input_shape_list": [(5, 3)], "num_calls": 1},
        )

        image_meta = [MixedImageMeta(dataset_name="ds", source_id=i) for i in range(5)]
        items = list(atomize(record, {"atom_mode": "chunk", "chunk_rows": 2}, image_meta, model_run_id=7))

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].x.shape[0], 2)
        self.assertEqual(items[0].meta["model_run_id"], 7)


if __name__ == "__main__":
    unittest.main()
