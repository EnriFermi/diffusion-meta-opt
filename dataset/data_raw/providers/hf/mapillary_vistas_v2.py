from __future__ import annotations

from typing import Any

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.providers.hf.virtual_dataset import HFVirtualDataset
from dataset.data_raw.registry import register_dataset

DATASET_NAME = "mapillary_vistas_v2"


def build_dataset(cfg: Any, global_data_root: str, seed: int, hf_cfg: Any) -> HFVirtualDataset:
    plain = to_plain_dict(cfg)
    plain["gated"] = True
    return HFVirtualDataset(cfg=plain, global_data_root=global_data_root, seed=seed, hf_cfg=hf_cfg)


register_dataset(DATASET_NAME, build_dataset)
