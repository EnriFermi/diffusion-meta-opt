from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from dataset.models.base_virtual_model import BaseVirtualModel
from dataset.models.model_runner import ModelRunner
from dataset.models.types import LayerIORecord


class DummyModel(BaseVirtualModel):
    def __init__(self, name: str) -> None:
        super().__init__(cfg={}, global_cfg={})
        self.name = name
        self.loaded = False
        self.unloaded = False
        self.batch_sizes: list[int] = []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True
        self.loaded = False

    def run(self, batch_pil):
        self.batch_sizes.append(len(batch_pil))
        n = len(batch_pil)
        return [
            LayerIORecord(
                model_name=self.name,
                layer_name="linear",
                weight=torch.ones(3, 4),
                inputs=torch.ones(n, 4),
                outputs=torch.ones(n, 3),
                meta={"num_calls": 1, "input_shape_list": [(n, 4)], "output_shape_list": [(n, 3)]},
            )
        ]

    def stats(self):
        return {"name": self.name, "loaded": self.loaded}


class TestModelRunner(unittest.TestCase):
    def test_microbatching_and_lru_eviction(self) -> None:
        global_cfg = {"data": {"max_loaded_models": 1}}
        model_cfgs = {
            "m1": {"batch_size": 2},
            "m2": {"batch_size": 2},
        }

        created: dict[str, DummyModel] = {}

        def _build(name, cfg, global_cfg):
            del cfg, global_cfg
            model = DummyModel(name)
            created[name] = model
            return model

        with patch("dataset.models.model_runner.create_model", side_effect=_build):
            runner = ModelRunner(global_cfg=global_cfg, model_cfgs=model_cfgs)

            records = runner.run_model_on_images("m1", [1, 2, 3, 4, 5])
            self.assertEqual(created["m1"].batch_sizes, [2, 2, 1])
            self.assertEqual(len(records), 1)
            self.assertEqual(tuple(records[0].inputs.shape), (5, 4))

            _ = runner.run_model_on_images("m2", [7])
            self.assertTrue(created["m1"].unloaded)
            self.assertTrue(created["m2"].loaded)

            runner.shutdown()
            self.assertTrue(created["m2"].unloaded)


if __name__ == "__main__":
    unittest.main()
