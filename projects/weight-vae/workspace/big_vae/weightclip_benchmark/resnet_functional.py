from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class ConvSpec:
    key: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    topological_depth: int = 0
    residual_role: str = "main"

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (self.out_channels, self.in_channels // self.groups, *self.kernel_size)


class ConvWeightProvider(Protocol):
    def get_conv_weight(self, spec: ConvSpec, activation_context: torch.Tensor) -> torch.Tensor: ...


class StaticConvWeightProvider:
    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        self.state = state

    def get_conv_weight(self, spec: ConvSpec, activation_context: torch.Tensor) -> torch.Tensor:
        del activation_context
        value = self.state[f"{spec.key}.weight"]
        if tuple(value.shape) != spec.weight_shape:
            raise ValueError(f"{spec.key} shape mismatch: {tuple(value.shape)} vs {spec.weight_shape}")
        return value


@dataclass(slots=True)
class FunctionalResNetConfig:
    width_mult: float = 0.5
    channels_in: int = 3
    num_classes: int = 10
    activation: str = "relu"
    context_rows: int = 512
    context_seed: int = 0
    head_policy: str = "original_frozen_source"
    bn_policy: str = "original_frozen_source"
    dropout: float = 0.0


def resnet18slim_conv_specs(config: FunctionalResNetConfig) -> tuple[ConvSpec, ...]:
    channels = [max(1, int(base * config.width_mult)) for base in (64, 128, 256, 512)]
    specs: list[ConvSpec] = [
        ConvSpec("conv1", config.channels_in, channels[0], (3, 3), (1, 1), (1, 1), topological_depth=0, residual_role="stem")
    ]
    in_channels = channels[0]
    depth = 1
    for stage, out_channels in enumerate(channels, start=1):
        for block in range(2):
            stride = 2 if stage > 1 and block == 0 else 1
            prefix = f"layer{stage}.{block}"
            specs.append(
                ConvSpec(
                    f"{prefix}.conv1",
                    in_channels,
                    out_channels,
                    (3, 3),
                    (stride, stride),
                    (1, 1),
                    topological_depth=depth,
                    residual_role="main_conv1",
                )
            )
            specs.append(
                ConvSpec(
                    f"{prefix}.conv2",
                    out_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    topological_depth=depth + 1,
                    residual_role="main_conv2",
                )
            )
            if stride != 1 or in_channels != out_channels:
                specs.append(
                    ConvSpec(
                        f"{prefix}.shortcut.0",
                        in_channels,
                        out_channels,
                        (1, 1),
                        (stride, stride),
                        (0, 0),
                        topological_depth=depth,
                        residual_role="shortcut",
                    )
                )
            in_channels = out_channels
            depth += 2
    return tuple(specs)


class FunctionalResNet18Slim(nn.Module):
    """Exact topological ResNet18Slim execution with generated convolution weights.

    The provider is called immediately before each operation, so activation-conditioned
    decoders see activations produced by the generated prefix. Context construction is
    explicitly detached; gradients reach the current layer's code through the decoder but
    never flow to earlier layers through the context side channel.
    """

    def __init__(
        self,
        config: FunctionalResNetConfig,
        provider: ConvWeightProvider,
        *,
        source_state: Mapping[str, torch.Tensor] | None = None,
        random_head_seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.provider = provider
        self.specs = {spec.key: spec for spec in resnet18slim_conv_specs(config)}
        self._source_state = dict(source_state or {})
        channels = max(1, int(512 * config.width_mult))
        head_weight, head_bias = self._build_head(channels, random_head_seed)
        self.register_buffer("head_weight", head_weight, persistent=True)
        self.register_buffer("head_bias", head_bias, persistent=True)
        self._register_bn_state()

    def _build_head(self, in_features: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        policy = self.config.head_policy
        if policy == "original_frozen_source":
            if "fc.weight" not in self._source_state or "fc.bias" not in self._source_state:
                raise ValueError("original_frozen_source head requires fc.weight and fc.bias")
            weight = self._source_state["fc.weight"].detach().clone()
            bias = self._source_state["fc.bias"].detach().clone()
        elif policy == "default_random_frozen":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                head = nn.Linear(in_features, self.config.num_classes)
            weight, bias = head.weight.detach().clone(), head.bias.detach().clone()
        else:
            raise ValueError(f"unsupported head_policy {policy!r}")
        if tuple(weight.shape) != (self.config.num_classes, in_features):
            raise ValueError(f"head shape mismatch: {tuple(weight.shape)}")
        return weight, bias

    def _register_bn_state(self) -> None:
        bn_names = ["bn1"]
        for stage in range(1, 5):
            for block in range(2):
                prefix = f"layer{stage}.{block}"
                bn_names.extend((f"{prefix}.bn1", f"{prefix}.bn2"))
                if f"{prefix}.shortcut.0" in self.specs:
                    bn_names.append(f"{prefix}.shortcut.1")
        for name in bn_names:
            channels = self._bn_channels(name)
            if self.config.bn_policy == "original_frozen_source":
                required = [f"{name}.{field}" for field in ("weight", "bias", "running_mean", "running_var")]
                missing = [key for key in required if key not in self._source_state]
                if missing:
                    raise ValueError(f"source BN policy missing {missing}")
                values = [self._source_state[key].detach().clone() for key in required]
            elif self.config.bn_policy == "default_frozen":
                values = [torch.ones(channels), torch.zeros(channels), torch.zeros(channels), torch.ones(channels)]
            else:
                raise ValueError(f"unsupported bn_policy {self.config.bn_policy!r}")
            safe = name.replace(".", "__")
            for field, value in zip(("weight", "bias", "running_mean", "running_var"), values, strict=True):
                self.register_buffer(f"bn__{safe}__{field}", value, persistent=True)

    def _bn_channels(self, name: str) -> int:
        if name == "bn1":
            return max(1, int(64 * self.config.width_mult))
        stage = int(name[len("layer")])
        return max(1, int((64 * (2 ** (stage - 1))) * self.config.width_mult))

    def _bn(self, x: torch.Tensor, name: str) -> torch.Tensor:
        safe = name.replace(".", "__")
        return F.batch_norm(
            x,
            getattr(self, f"bn__{safe}__running_mean"),
            getattr(self, f"bn__{safe}__running_var"),
            getattr(self, f"bn__{safe}__weight"),
            getattr(self, f"bn__{safe}__bias"),
            training=False,
            momentum=0.0,
            eps=1e-5,
        )

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.activation == "relu":
            return F.relu(x)
        if self.config.activation == "gelu":
            return F.gelu(x)
        if self.config.activation == "silu":
            return F.silu(x)
        if self.config.activation == "tanh":
            return torch.tanh(x)
        raise ValueError(f"unsupported activation {self.config.activation!r}")

    def _context(self, x: torch.Tensor, spec: ConvSpec) -> torch.Tensor:
        # This detach is the contract: no decoder-context gradient to the prefix.
        unfolded = F.unfold(
            x.detach(),
            kernel_size=spec.kernel_size,
            dilation=spec.dilation,
            padding=spec.padding,
            stride=spec.stride,
        ).transpose(1, 2).reshape(-1, spec.in_channels * spec.kernel_size[0] * spec.kernel_size[1])
        max_rows = int(self.config.context_rows)
        if max_rows <= 0 or unfolded.shape[0] <= max_rows:
            return unfolded.contiguous()
        digest = hashlib.sha256(f"{self.config.context_seed}:{spec.key}".encode()).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        index = torch.randperm(unfolded.shape[0], generator=generator)[:max_rows].to(unfolded.device)
        return unfolded[index].contiguous()

    def _conv(self, x: torch.Tensor, key: str) -> torch.Tensor:
        spec = self.specs[key]
        cache_active = bool(getattr(self.provider, "cache_active", False))
        context = x.new_empty((0, spec.in_channels * spec.kernel_size[0] * spec.kernel_size[1])) if cache_active else self._context(x, spec)
        weight = self.provider.get_conv_weight(spec, context)
        if tuple(weight.shape) != spec.weight_shape:
            raise ValueError(f"provider returned {tuple(weight.shape)} for {key}, expected {spec.weight_shape}")
        return F.conv2d(x, weight, None, spec.stride, spec.padding, spec.dilation, spec.groups)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._activation(self._bn(self._conv(images, "conv1"), "bn1"))
        for stage in range(1, 5):
            for block in range(2):
                prefix = f"layer{stage}.{block}"
                residual = x
                main = self._activation(self._bn(self._conv(x, f"{prefix}.conv1"), f"{prefix}.bn1"))
                main = self._bn(self._conv(main, f"{prefix}.conv2"), f"{prefix}.bn2")
                shortcut_key = f"{prefix}.shortcut.0"
                if shortcut_key in self.specs:
                    residual = self._bn(self._conv(residual, shortcut_key), f"{prefix}.shortcut.1")
                x = self._activation(main + residual)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        return F.linear(x, self.head_weight, self.head_bias)


def coverage_against_state_dict(config: FunctionalResNetConfig, state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    generated = {f"{spec.key}.weight" for spec in resnet18slim_conv_specs(config)}
    coverage: dict[str, str] = {}
    for key in state:
        if key in generated:
            coverage[key] = "generated"
        elif key.startswith("fc."):
            coverage[key] = "head"
        elif ".bn" in f".{key}" or key.startswith("bn1.") or ".shortcut.1." in key:
            coverage[key] = "batchnorm"
        else:
            coverage[key] = "unknown"
    return coverage
