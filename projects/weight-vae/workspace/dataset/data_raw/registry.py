from __future__ import annotations

from typing import Any, Callable

from dataset.data_raw.core.base_virtual_dataset import BaseVirtualDataset

DatasetBuilder = Callable[[Any, str, int, Any], BaseVirtualDataset]

_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str, builder: DatasetBuilder) -> None:
    key = str(name)
    if key in _REGISTRY:
        raise ValueError(f"Dataset builder already registered for '{key}'")
    _REGISTRY[key] = builder


def create_dataset(name: str, cfg: Any, global_root: str, seed: int, hf_cfg: Any = None) -> BaseVirtualDataset:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Registered datasets: {available}")

    builder = _REGISTRY[name]
    return builder(cfg, global_root, seed, hf_cfg)


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())
