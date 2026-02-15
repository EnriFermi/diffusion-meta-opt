from __future__ import annotations

import unittest
from copy import deepcopy

from dataset.shared.collector_service import CollectorService


def _base_cfg(value) -> dict:
    return {
        "hf": {"token": "hf_dummy"},
        "data": {
            "path": "./data",
            "seed": 1,
            "dataset_config_dirs": ["conf/data/datasets"],
            "enabled_datasets": ["coco2017"],
            "dataset_overrides": {
                "coco2017": {
                    "models": ["clip_vit_b32"],
                },
            },
        },
        "train": {"device": "cpu"},
        "collector": {
            "mode": "interleaved",
            "device": None,
            "layer_output_splitting": {"xy_samples_random_slice": value},
        },
        "models": {"model_config_dirs": ["conf/data/models"]},
        "streaming": {"mode": "none"},
    }


class TestCollectorConfigValidation(unittest.TestCase):
    def test_missing_field_fails(self) -> None:
        cfg = _base_cfg(8)
        del cfg["collector"]["layer_output_splitting"]["xy_samples_random_slice"]
        with self.assertRaisesRegex(ValueError, "xy_samples_random_slice must be a positive integer"):
            CollectorService(cfg)

    def test_none_fails(self) -> None:
        cfg = _base_cfg(None)
        with self.assertRaisesRegex(ValueError, "xy_samples_random_slice must be a positive integer"):
            CollectorService(cfg)

    def test_invalid_string_fails(self) -> None:
        cfg = _base_cfg("abc")
        with self.assertRaisesRegex(ValueError, "xy_samples_random_slice must be a positive integer"):
            CollectorService(cfg)

    def test_zero_and_negative_fail(self) -> None:
        for v in [0, -1]:
            cfg = _base_cfg(v)
            with self.assertRaisesRegex(ValueError, "xy_samples_random_slice must be a positive integer"):
                CollectorService(cfg)

    def test_valid_positive_int_passes_constructor(self) -> None:
        cfg = _base_cfg(16)
        service = CollectorService(deepcopy(cfg))
        self.assertEqual(service.atom_cfg["xy_samples_random_slice"], 16)


if __name__ == "__main__":
    unittest.main()

