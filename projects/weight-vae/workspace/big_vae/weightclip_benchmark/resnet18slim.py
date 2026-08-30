"""Exact local copy of the WeightCLIP ResNet18Slim architecture contract.

The implementation follows HSG-AIML/weightCLIP commit
``be080677a6eceacdbe3b2823caffe3c0cc73fa7e``.  It is kept local so zoo
creation and checkpoint validation do not depend on an editable SANE install.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def activation_layer(name: str) -> nn.Module:
    factories: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return factories[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name!r}") from exc


def initialize_weights(module: nn.Module, initialization: str) -> None:
    """Match the released recursive initializer, including bias=0.01."""

    for child in module.modules():
        if not isinstance(child, (nn.Linear, nn.Conv2d)):
            continue
        if initialization == "xavier_uniform":
            nn.init.xavier_uniform_(child.weight)
        elif initialization == "xavier_normal":
            nn.init.xavier_normal_(child.weight)
        elif initialization == "kaiming_uniform":
            nn.init.kaiming_uniform_(child.weight, nonlinearity="relu")
        elif initialization == "kaiming_normal":
            nn.init.kaiming_normal_(child.weight, nonlinearity="relu")
        elif initialization == "uniform":
            nn.init.uniform_(child.weight)
        elif initialization == "normal":
            nn.init.normal_(child.weight, std=0.05)
        else:
            raise ValueError(f"Unknown initialization method: {initialization}")
        if child.bias is not None:
            nn.init.constant_(child.bias, 0.01)


class BasicBlockCustom(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, nlin: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = activation_layer(nlin)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = activation_layer(nlin)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class ResNet18Slim(nn.Module):
    """WeightCLIP half-width ResNet-18 (32/64/128/256 channels at 0.5x)."""

    def __init__(
        self,
        channels_in: int = 3,
        o_dim: int = 10,
        nlin: str = "relu",
        dropout: float = 0.0,
        init_type: str | None = "kaiming_uniform",
        width_mult: float = 0.5,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (max(1, int(base * width_mult)) for base in (64, 128, 256, 512))
        self.width_mult = float(width_mult)
        self.channels = (c1, c2, c3, c4)
        self.conv1 = nn.Conv2d(channels_in, c1, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = activation_layer(nlin)
        self.layer1 = self._make_layer(c1, c1, 2, 1, nlin)
        self.layer2 = self._make_layer(c1, c2, 2, 2, nlin)
        self.layer3 = self._make_layer(c2, c3, 2, 2, nlin)
        self.layer4 = self._make_layer(c3, c4, 2, 2, nlin)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c4, o_dim)
        if init_type is not None:
            initialize_weights(self, init_type)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int, nlin: str) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlockCustom(in_channels, out_channels, stride, nlin)]
        layers.extend(BasicBlockCustom(out_channels, out_channels, 1, nlin) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


OFFICIAL_WEIGHTCLIP_COMMIT = "be080677a6eceacdbe3b2823caffe3c0cc73fa7e"

