from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.nn.functional as F

from MetaOpt.cifar10 import Cifar10EpisodeSampler
from MetaOpt.models import TinyResNet, TwoLayerMLP
from MetaOpt.tasks import build_task_sampler


@dataclass
class EpisodeBatch:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor


class EpisodeSampler(Protocol):
    def sample(self, device: torch.device) -> EpisodeBatch: ...


class RegressionEpisodeSampler:
    def __init__(self, cfg):
        self.task_mixer = build_task_sampler(cfg)

    def sample(self, device: torch.device) -> EpisodeBatch:
        task = self.task_mixer.sample_task()
        x_train, y_train, x_val, y_val = task.sample(device)
        return EpisodeBatch(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)


class Cifar10SamplerAdapter:
    def __init__(self, cfg):
        self.sampler = Cifar10EpisodeSampler(
            data_dir=cfg.cifar10.data_dir,
            download=bool(cfg.cifar10.download),
            n_train=int(cfg.cifar10.n_train),
            n_val=int(cfg.cifar10.n_val),
            augment=bool(cfg.cifar10.augment),
            normalize=bool(cfg.cifar10.normalize),
            pin_memory=bool(cfg.perf.pin_memory),
            num_workers=int(cfg.perf.num_workers),
            persistent_workers=bool(cfg.perf.persistent_workers),
            cache_on_gpu=bool(cfg.cifar10.cache_on_gpu),
        )

    def sample(self, device: torch.device) -> EpisodeBatch:
        b = self.sampler.sample(device)
        return EpisodeBatch(x_train=b.x_train, y_train=b.y_train, x_val=b.x_val, y_val=b.y_val)


def build_episode_sampler(cfg) -> EpisodeSampler:
    name = str(cfg.downstream.name)
    if name == "regression":
        return RegressionEpisodeSampler(cfg)
    if name == "cifar10":
        return Cifar10SamplerAdapter(cfg)
    raise ValueError(f"Unknown downstream.name={name!r}")


def build_downstream_model(cfg):
    name = str(cfg.downstream.name)
    if name == "regression":
        return TwoLayerMLP(cfg.model.hidden)
    if name == "cifar10":
        resnet_cfg = cfg.model.resnet
        return TinyResNet(
            base_channels=int(resnet_cfg.base_channels),
            blocks=tuple(int(x) for x in resnet_cfg.blocks),
            num_classes=int(resnet_cfg.num_classes),
        )
    raise ValueError(f"Unknown downstream.name={name!r}")


def build_loss_fn(cfg) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = str(cfg.downstream.name)
    if name == "regression":
        return F.mse_loss
    if name == "cifar10":
        return F.cross_entropy
    raise ValueError(f"Unknown downstream.name={name!r}")
