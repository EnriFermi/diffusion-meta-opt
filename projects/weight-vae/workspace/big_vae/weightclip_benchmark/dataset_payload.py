"""Stable tensor-backed dataset payload used by WeightCLIP zoo artifacts.

The first production materialization was launched through ``python -m`` and
therefore pickled this class as ``__main__.CachedDataset``.  Keep the class in
an importable module for all new artifacts; :func:`load_cached_dataset_payload`
also provides a narrowly-scoped compatibility bridge for those immutable v1
files.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Tensor-backed split with the attribute interface used by WeightCLIP."""

    def __init__(self, data: torch.Tensor | None = None, targets: torch.Tensor | None = None) -> None:
        self.transform = None
        self.data = torch.empty(0, 3, 32, 32) if data is None else data
        self.targets = torch.empty(0, dtype=torch.long) if targets is None else targets

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index]
        return (self.transform(image) if self.transform is not None else image), self.targets[index]

    def batches(self, batch_size: int, *, shuffle: bool, device: str) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        order = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for start in range(0, len(self), batch_size):
            indices = order[start : start + batch_size]
            yield (
                self.data[indices].to(device, non_blocking=True),
                self.targets[indices].to(device, non_blocking=True),
            )


@contextmanager
def _legacy_main_pickle_alias() -> Iterator[None]:
    """Temporarily resolve immutable v1 ``__main__.CachedDataset`` pickles."""

    main_module = sys.modules.get("__main__")
    if main_module is None:
        yield
        return
    sentinel = object()
    previous = getattr(main_module, "CachedDataset", sentinel)
    setattr(main_module, "CachedDataset", CachedDataset)
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(main_module, "CachedDataset")
        else:
            setattr(main_module, "CachedDataset", previous)


def load_cached_dataset_payload(
    path: str,
    *,
    map_location: str = "cpu",
    splits: Iterable[str] | None = None,
) -> dict[str, CachedDataset]:
    """Load either stable v2 or immutable legacy v1 payload, then normalize it.

    The compatibility alias exists only during unpickling.  Returned objects
    always have the stable module-qualified type and contiguous CPU tensors.
    """

    with _legacy_main_pickle_alias():
        payload: Any = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"dataset payload must be a mapping: {path}")
    required = {"trainset", "valset", "testset"}
    if set(payload) != required:
        raise ValueError(f"dataset payload must contain exactly {sorted(required)}: {path}")
    selected = required if splits is None else {str(split) for split in splits}
    if not selected or not selected <= required:
        raise ValueError(f"requested unsupported dataset splits: {sorted(selected - required)}")
    result: dict[str, CachedDataset] = {}
    for split in sorted(selected):
        source = payload[split]
        data = getattr(source, "data", None)
        targets = getattr(source, "targets", None)
        if not torch.is_tensor(data) or not torch.is_tensor(targets):
            raise TypeError(f"dataset payload {split} lacks tensor data/targets: {path}")
        result[split] = CachedDataset(
            data.detach().to(device="cpu").contiguous(),
            targets.detach().to(device="cpu", dtype=torch.long).contiguous(),
        )
    return result
