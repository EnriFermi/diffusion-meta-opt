from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from post_train_research.vit_latent_scaling.config import DataConfig, ModelConfig


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _repeat_channel_stats(values: tuple[float, ...], *, count: int) -> tuple[float, ...]:
    if len(values) == count:
        return tuple(float(item) for item in values)
    if len(values) == 1:
        return tuple(float(values[0]) for _ in range(int(count)))
    raise ValueError(f"cannot adapt stats of length {len(values)} to channel count {count}")


def maybe_subset(dataset: Any, limit: int) -> Any:
    if int(limit) <= 0:
        return dataset
    return Subset(dataset, list(range(min(int(limit), len(dataset)))))


def _make_loaders(data_cfg: DataConfig, device: torch.device, train_set: Any, test_set: Any, *, seed: int) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=int(data_cfg.batch_size),
        shuffle=True,
        num_workers=int(data_cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(data_cfg.num_workers) > 0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(data_cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(data_cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(data_cfg.num_workers) > 0,
    )
    return train_loader, test_loader


def build_mnist_loaders(data_cfg: DataConfig, model_cfg: ModelConfig, device: torch.device, *, seed: int) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    train_steps = []
    eval_steps = []
    if int(model_cfg.image_size) != 28:
        resize = transforms.Resize((int(model_cfg.image_size), int(model_cfg.image_size)))
        train_steps.append(resize)
        eval_steps.append(resize)
    if int(model_cfg.in_channels) == 3:
        train_steps.append(transforms.Grayscale(num_output_channels=3))
        eval_steps.append(transforms.Grayscale(num_output_channels=3))
    elif int(model_cfg.in_channels) != 1:
        raise ValueError(f"MNIST supports only in_channels 1 or 3, got {model_cfg.in_channels}")
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                _repeat_channel_stats(MNIST_MEAN, count=int(model_cfg.in_channels)),
                _repeat_channel_stats(MNIST_STD, count=int(model_cfg.in_channels)),
            ),
        ]
    )
    eval_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                _repeat_channel_stats(MNIST_MEAN, count=int(model_cfg.in_channels)),
                _repeat_channel_stats(MNIST_STD, count=int(model_cfg.in_channels)),
            ),
        ]
    )
    root = str(Path(data_cfg.data_dir).expanduser())
    train_set = datasets.MNIST(root=root, train=True, download=bool(data_cfg.download), transform=transforms.Compose(train_steps))
    test_set = datasets.MNIST(root=root, train=False, download=bool(data_cfg.download), transform=transforms.Compose(eval_steps))
    return _make_loaders(
        data_cfg,
        device,
        maybe_subset(train_set, data_cfg.train_subset),
        maybe_subset(test_set, data_cfg.test_subset),
        seed=seed,
    )


def build_cifar10_loaders(data_cfg: DataConfig, model_cfg: ModelConfig, device: torch.device, *, seed: int) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    train_steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    eval_steps = []
    if int(model_cfg.in_channels) == 1:
        train_steps.append(transforms.Grayscale(num_output_channels=1))
        eval_steps.append(transforms.Grayscale(num_output_channels=1))
    elif int(model_cfg.in_channels) != 3:
        raise ValueError(f"CIFAR-10 supports only in_channels 1 or 3, got {model_cfg.in_channels}")
    if int(model_cfg.image_size) != 32:
        resize = transforms.Resize((int(model_cfg.image_size), int(model_cfg.image_size)))
        train_steps.append(resize)
        eval_steps.append(resize)
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                _repeat_channel_stats(CIFAR10_MEAN, count=int(model_cfg.in_channels)),
                _repeat_channel_stats(CIFAR10_STD, count=int(model_cfg.in_channels)),
            ),
        ]
    )
    eval_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                _repeat_channel_stats(CIFAR10_MEAN, count=int(model_cfg.in_channels)),
                _repeat_channel_stats(CIFAR10_STD, count=int(model_cfg.in_channels)),
            ),
        ]
    )
    root = str(Path(data_cfg.data_dir).expanduser())
    train_set = datasets.CIFAR10(root=root, train=True, download=bool(data_cfg.download), transform=transforms.Compose(train_steps))
    test_set = datasets.CIFAR10(root=root, train=False, download=bool(data_cfg.download), transform=transforms.Compose(eval_steps))
    return _make_loaders(
        data_cfg,
        device,
        maybe_subset(train_set, data_cfg.train_subset),
        maybe_subset(test_set, data_cfg.test_subset),
        seed=seed,
    )


def build_imagenet_loaders(data_cfg: DataConfig, model_cfg: ModelConfig, device: torch.device, *, seed: int) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    eval_resize = max(int(model_cfg.image_size), int(round(float(model_cfg.image_size) * 256.0 / 224.0)))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(int(model_cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(int(model_cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    root = Path(data_cfg.data_dir).expanduser()
    train_dir = root / "train"
    val_dir = root / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        train_set = datasets.ImageFolder(str(train_dir), transform=train_transform)
        test_set = datasets.ImageFolder(str(val_dir), transform=eval_transform)
    else:
        train_set = datasets.ImageNet(root=str(root), split="train", transform=train_transform)
        test_set = datasets.ImageNet(root=str(root), split="val", transform=eval_transform)
    return _make_loaders(
        data_cfg,
        device,
        maybe_subset(train_set, data_cfg.train_subset),
        maybe_subset(test_set, data_cfg.test_subset),
        seed=seed,
    )


def build_loaders(data_cfg: DataConfig, model_cfg: ModelConfig, device: torch.device, *, seed: int) -> tuple[DataLoader, DataLoader]:
    dataset = str(data_cfg.dataset).strip().lower()
    if dataset == "mnist":
        return build_mnist_loaders(data_cfg, model_cfg, device, seed=seed)
    if dataset == "cifar10":
        return build_cifar10_loaders(data_cfg, model_cfg, device, seed=seed)
    if dataset == "imagenet":
        return build_imagenet_loaders(data_cfg, model_cfg, device, seed=seed)
    raise ValueError(f"Unsupported dataset: {data_cfg.dataset!r}")
