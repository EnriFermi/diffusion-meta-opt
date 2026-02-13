from __future__ import annotations

from typing import Any

from dataset.data_raw.providers.hf.virtual_dataset import HFVirtualDataset
from dataset.data_raw.registry import register_dataset

DATASET_NAME = "visual_genome"


def build_dataset(cfg: Any, global_data_root: str, seed: int, hf_cfg: Any) -> HFVirtualDataset:
    return HFVirtualDataset(cfg=cfg, global_data_root=global_data_root, seed=seed, hf_cfg=hf_cfg)


register_dataset(DATASET_NAME, build_dataset)
