from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dataset.models.hooks import attach_hooks, detach_hooks, flatten_to_2d


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(4, 3, bias=False)
        self.proj2 = nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.proj1(inputs)
        return self.proj2(hidden)


class TestModelHooks(unittest.TestCase):
    def test_flatten_to_2d(self) -> None:
        tensor = torch.randn(2, 3, 4)
        flattened = flatten_to_2d(tensor)
        self.assertEqual(tuple(flattened.shape), (6, 4))

    def test_linear_hooks_collect_and_cap(self) -> None:
        model = ToyModel()
        handles, buffers = attach_hooks(model, cfg={"max_records_per_layer": 5})

        try:
            inputs = torch.randn(2, 3, 4)
            _ = model(inputs)
        finally:
            detach_hooks(handles)

        self.assertIn("proj1", buffers)
        self.assertIn("proj2", buffers)

        proj1_inputs = torch.cat(buffers["proj1"]["inputs"], dim=0)
        proj1_outputs = torch.cat(buffers["proj1"]["outputs"], dim=0)

        self.assertEqual(tuple(proj1_inputs.shape), (5, 4))
        self.assertEqual(tuple(proj1_outputs.shape), (5, 3))

        self.assertGreaterEqual(buffers["proj1"]["num_calls"], 1)

    def test_hook_filters_and_max_layers(self) -> None:
        model = ToyModel()
        handles, buffers = attach_hooks(
            model,
            cfg={
                "include_regex": r"^proj",
                "exclude_regex": r"proj2",
                "max_layers": 1,
            },
        )

        try:
            _ = model(torch.randn(2, 4))
        finally:
            detach_hooks(handles)

        self.assertIn("proj1", buffers)
        self.assertNotIn("proj2", buffers)
        self.assertEqual(len(buffers), 1)


if __name__ == "__main__":
    unittest.main()
