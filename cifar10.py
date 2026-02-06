from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _normalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_CIFAR10_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR10_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _random_crop(x: torch.Tensor, out_hw: int = 32, pad: int = 4) -> torch.Tensor:
    if pad <= 0:
        return x
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    b, c, h, w = x.shape
    if h == out_hw and w == out_hw:
        return x
    max_y = h - out_hw
    max_x = w - out_hw
    off_y = torch.randint(0, max_y + 1, (b,), device=x.device)
    off_x = torch.randint(0, max_x + 1, (b,), device=x.device)
    yy = off_y[:, None] + torch.arange(out_hw, device=x.device)[None, :]
    xx = off_x[:, None] + torch.arange(out_hw, device=x.device)[None, :]
    bb = torch.arange(b, device=x.device)[:, None, None]
    return x[bb, :, yy[:, :, None], xx[:, None, :]]


def _random_hflip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if p <= 0:
        return x
    b = x.size(0)
    mask = torch.rand(b, device=x.device) < p
    if mask.any():
        x = x.clone()
        x[mask] = x[mask].flip(-1)
    return x


@dataclass
class Cifar10EpisodeBatch:
    x_train: torch.Tensor  # (n_train, 3, 32, 32)
    y_train: torch.Tensor  # (n_train,)
    x_val: torch.Tensor    # (n_val, 3, 32, 32)
    y_val: torch.Tensor    # (n_val,)


class Cifar10EpisodeSampler:
    """
    Samples small (train,val) subsets from CIFAR-10 for each episode.

    This is intentionally lightweight: it materializes CIFAR-10 once as CPU tensors and
    returns per-episode tensors moved to the requested device.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        download: bool = True,
        n_train: int = 64,
        n_val: int = 64,
        augment: bool = True,
        normalize: bool = True,
        pin_memory: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = True,
        cache_on_gpu: bool = False,
    ):
        try:
            from torchvision.datasets import CIFAR10  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "CIFAR-10 downstream requires torchvision. Install it (pip/conda) and retry."
            ) from e

        train_ds = CIFAR10(root=data_dir, train=True, download=download)
        test_ds = CIFAR10(root=data_dir, train=False, download=download)

        self.train_x = torch.from_numpy(train_ds.data).permute(0, 3, 1, 2).contiguous()  # uint8
        self.train_y = torch.tensor(train_ds.targets, dtype=torch.long)
        self.val_x = torch.from_numpy(test_ds.data).permute(0, 3, 1, 2).contiguous()  # uint8
        self.val_y = torch.tensor(test_ds.targets, dtype=torch.long)

        if pin_memory and torch.cuda.is_available():
            self.train_x = self.train_x.pin_memory()
            self.train_y = self.train_y.pin_memory()
            self.val_x = self.val_x.pin_memory()
            self.val_y = self.val_y.pin_memory()

        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.cache_on_gpu = bool(cache_on_gpu)

        self.num_workers = int(num_workers)
        self.persistent_workers = bool(persistent_workers)
        self._use_dataloader = (self.num_workers > 0) and (not self.cache_on_gpu)
        self._train_loader = None
        self._val_loader = None
        self._train_iter = None
        self._val_iter = None
        if self._use_dataloader:
            from torch.utils.data import DataLoader, TensorDataset

            train_ds = TensorDataset(self.train_x, self.train_y)
            val_ds = TensorDataset(self.val_x, self.val_y)
            self._train_loader = DataLoader(
                train_ds,
                batch_size=self.n_train,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=self.persistent_workers,
            )
            self._val_loader = DataLoader(
                val_ds,
                batch_size=self.n_val,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=self.persistent_workers,
            )
            self._train_iter = iter(self._train_loader)
            self._val_iter = iter(self._val_loader)

        self._cached_device: Optional[torch.device] = None

    def _maybe_cache(self, device: torch.device):
        if not self.cache_on_gpu:
            return
        if self._cached_device == device:
            return
        if device.type != "cuda":
            return
        self.train_x = self.train_x.to(device, non_blocking=True)
        self.train_y = self.train_y.to(device, non_blocking=True)
        self.val_x = self.val_x.to(device, non_blocking=True)
        self.val_y = self.val_y.to(device, non_blocking=True)
        self._cached_device = device

    @torch.no_grad()
    def sample(self, device: torch.device) -> Cifar10EpisodeBatch:
        device = torch.device(device)
        self._maybe_cache(device)

        if self._use_dataloader:
            assert self._train_loader is not None and self._val_loader is not None
            assert self._train_iter is not None and self._val_iter is not None
            try:
                x_train, y_train = next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self._train_loader)
                x_train, y_train = next(self._train_iter)
            try:
                x_val, y_val = next(self._val_iter)
            except StopIteration:
                self._val_iter = iter(self._val_loader)
                x_val, y_val = next(self._val_iter)
        else:
            train_idx = torch.randint(0, self.train_x.size(0), (self.n_train,), device=self.train_x.device)
            val_idx = torch.randint(0, self.val_x.size(0), (self.n_val,), device=self.val_x.device)
            x_train = self.train_x[train_idx]
            y_train = self.train_y[train_idx]
            x_val = self.val_x[val_idx]
            y_val = self.val_y[val_idx]

        if x_train.device != device:
            x_train = x_train.to(device, non_blocking=True)
            y_train = y_train.to(device, non_blocking=True)
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

        x_train = x_train.float().div_(255.0)
        x_val = x_val.float().div_(255.0)

        if self.augment:
            x_train = _random_hflip(x_train, p=0.5)
            x_train = _random_crop(x_train, out_hw=32, pad=4)

        if self.normalize:
            x_train = _normalize_cifar10(x_train)
            x_val = _normalize_cifar10(x_val)

        return Cifar10EpisodeBatch(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
