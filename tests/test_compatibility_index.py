from __future__ import annotations

import unittest

from dataset.shared.compatibility_index import CompatibilityIndex


class TestCompatibilityIndex(unittest.TestCase):
    def test_build_model_to_datasets_mapping(self) -> None:
        cfg = {
            "hf": {"token": "hf_dummy"},
            "data": {
                "path": "./data",
                "seed": 1,
                "dataset_config_dirs": ["conf/data/datasets"],
                "enabled_datasets": ["cc12m", "coco2017"],
                "dataset_overrides": {
                    "cc12m": {"models": []},
                    "coco2017": {"models": []},
                },
            },
            "train": {"device": "cuda:0"},
            "collector": {"mode": "interleaved", "device": None},
            "models": {
                "model_config_dirs": ["conf/data/models"],
            },
        }

        index = CompatibilityIndex(cfg)

        self.assertIn("clip_vit_b32", index.get_models())
        self.assertIn("cc12m", index.get_datasets_for_model("clip_vit_b32"))
        self.assertIn("coco2017", index.get_datasets_for_model("clip_vit_b32"))

    def test_async_filters_dataset_with_mismatched_collector_device(self) -> None:
        cfg = {
            "hf": {"token": "hf_dummy"},
            "data": {
                "path": "./data",
                "seed": 1,
                "dataset_config_dirs": ["conf/data/datasets"],
                "enabled_datasets": ["coco2017"],
                "dataset_overrides": {
                    "coco2017": {
                        "collector_device": "cuda:2",
                    },
                },
            },
            "train": {"device": "cuda:0"},
            "collector": {"mode": "async", "device": "cuda:1"},
            "models": {
                "model_config_dirs": ["conf/data/models"],
            },
        }

        index = CompatibilityIndex(cfg)
        self.assertEqual(index.get_models(), [])


if __name__ == "__main__":
    unittest.main()
