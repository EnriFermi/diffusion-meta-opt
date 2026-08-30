"""Exact operator-matrix adapters, tiling, coverage, and ResNet gauge views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    key: str
    module_name: str
    operation: str
    matrix_shape: tuple[int, int]
    native_shape: tuple[int, ...]
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    depth_index: int
    depth_normalized: float
    stage: int
    block: int
    role: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TileSpec:
    row_start: int
    col_start: int
    valid_rows: int
    valid_cols: int
    tile_rows: int
    tile_cols: int


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(int(v) for v in value)


def _role(name: str) -> tuple[int, int, str]:
    if name == "conv1":
        return 0, -1, "stem"
    parts = name.split(".")
    if len(parts) >= 3 and parts[0].startswith("layer"):
        stage = int(parts[0][5:])
        block = int(parts[1])
        suffix = ".".join(parts[2:])
        if suffix == "shortcut.0":
            return stage, block, "residual_projection"
        if suffix == "conv1":
            return stage, block, "residual_conv1"
        if suffix == "conv2":
            return stage, block, "residual_conv2"
    if name == "fc":
        return 5, -1, "classifier_head"
    return -1, -1, "unknown"


def supported_operator_specs(model: nn.Module, *, include_head: bool = False) -> list[OperatorSpec]:
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or (include_head and isinstance(module, nn.Linear)):
            if name == "fc" and not include_head:
                continue
            candidates.append((name, module))
    total = max(1, len(candidates) - 1)
    specs: list[OperatorSpec] = []
    for depth, (name, module) in enumerate(candidates):
        stage, block, role = _role(name)
        if isinstance(module, nn.Conv2d):
            kh, kw = _pair(module.kernel_size)
            if module.groups != 1:
                raise ValueError(f"Grouped convolution is not supported by the primary adapter: {name}")
            matrix_shape = (module.in_channels * kh * kw, module.out_channels)
            specs.append(
                OperatorSpec(
                    key=f"{name}.weight",
                    module_name=name,
                    operation="conv2d",
                    matrix_shape=matrix_shape,
                    native_shape=tuple(module.weight.shape),
                    kernel=(kh, kw),
                    stride=_pair(module.stride),
                    padding=_pair(module.padding),
                    dilation=_pair(module.dilation),
                    groups=int(module.groups),
                    depth_index=depth,
                    depth_normalized=float(depth / total),
                    stage=stage,
                    block=block,
                    role=role,
                )
            )
        elif isinstance(module, nn.Linear):
            specs.append(
                OperatorSpec(
                    key=f"{name}.weight",
                    module_name=name,
                    operation="linear",
                    matrix_shape=(module.in_features, module.out_features),
                    native_shape=tuple(module.weight.shape),
                    kernel=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    groups=1,
                    depth_index=depth,
                    depth_normalized=float(depth / total),
                    stage=stage,
                    block=block,
                    role=role,
                )
            )
    return specs


def parameter_to_matrix(parameter: torch.Tensor, spec: OperatorSpec) -> torch.Tensor:
    """Return the exact ``[d_in, d_out]`` operator matrix."""

    if tuple(parameter.shape) != spec.native_shape:
        raise ValueError(f"{spec.key}: expected {spec.native_shape}, got {tuple(parameter.shape)}")
    if spec.operation == "conv2d":
        # [out, in, kh, kw] -> [in * kh * kw, out], matching F.unfold order.
        return parameter.permute(1, 2, 3, 0).contiguous().reshape(spec.matrix_shape)
    if spec.operation == "linear":
        return parameter.transpose(0, 1).contiguous()
    raise ValueError(f"Unknown operation {spec.operation!r}")


def matrix_to_parameter(matrix: torch.Tensor, spec: OperatorSpec) -> torch.Tensor:
    if tuple(matrix.shape) != spec.matrix_shape:
        raise ValueError(f"{spec.key}: expected matrix {spec.matrix_shape}, got {tuple(matrix.shape)}")
    if spec.operation == "conv2d":
        in_channels = spec.native_shape[1]
        out_channels = spec.native_shape[0]
        kh, kw = spec.kernel
        return matrix.reshape(in_channels, kh, kw, out_channels).permute(3, 0, 1, 2).contiguous()
    if spec.operation == "linear":
        return matrix.transpose(0, 1).contiguous()
    raise ValueError(f"Unknown operation {spec.operation!r}")


def tile_matrix(matrix: torch.Tensor, tile_rows: int, tile_cols: int) -> list[tuple[torch.Tensor, torch.Tensor, TileSpec]]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank 2")
    result: list[tuple[torch.Tensor, torch.Tensor, TileSpec]] = []
    rows, cols = matrix.shape
    for row_start in range(0, rows, tile_rows):
        for col_start in range(0, cols, tile_cols):
            valid_rows = min(tile_rows, rows - row_start)
            valid_cols = min(tile_cols, cols - col_start)
            tile = matrix.new_zeros((tile_rows, tile_cols))
            mask = torch.zeros((tile_rows, tile_cols), dtype=torch.bool, device=matrix.device)
            tile[:valid_rows, :valid_cols] = matrix[row_start : row_start + valid_rows, col_start : col_start + valid_cols]
            mask[:valid_rows, :valid_cols] = True
            result.append((tile, mask, TileSpec(row_start, col_start, valid_rows, valid_cols, tile_rows, tile_cols)))
    return result


def untile_matrix(tiles: list[tuple[torch.Tensor, TileSpec]], shape: tuple[int, int]) -> torch.Tensor:
    if not tiles:
        raise ValueError("tiles must be nonempty")
    output = tiles[0][0].new_zeros(shape)
    coverage = torch.zeros(shape, dtype=torch.int16, device=output.device)
    for tile, spec in tiles:
        rs, cs = spec.row_start, spec.col_start
        output[rs : rs + spec.valid_rows, cs : cs + spec.valid_cols] = tile[: spec.valid_rows, : spec.valid_cols]
        coverage[rs : rs + spec.valid_rows, cs : cs + spec.valid_cols] += 1
    if not torch.all(coverage == 1):
        raise ValueError("tiles do not cover the matrix exactly once")
    return output


def state_key_coverage(model: nn.Module) -> dict[str, str]:
    """Classify every state key; unknown keys are a hard error at bank build."""

    generated = {spec.key for spec in supported_operator_specs(model)}
    coverage: dict[str, str] = {}
    for key in model.state_dict():
        if key in generated:
            coverage[key] = "generated"
        elif key.startswith("fc."):
            coverage[key] = "default_classifier_head"
        elif ".bn" in key or key.startswith("bn1.") or ".shortcut.1." in key:
            coverage[key] = "default_batchnorm"
        else:
            coverage[key] = "unknown"
    return coverage


def assert_complete_coverage(model: nn.Module) -> dict[str, str]:
    coverage = state_key_coverage(model)
    unknown = sorted(key for key, policy in coverage.items() if policy == "unknown")
    if unknown:
        raise ValueError(f"Unclassified state-dict keys: {unknown}")
    return coverage


def resnet18slim_permutation_axes() -> dict[str, tuple[str | None, ...]]:
    """Axis-to-gauge map matching WeightCLIP's released git-re-basin spec."""

    axes: dict[str, tuple[str | None, ...]] = {}

    def conv(name: str, pin: str | None, pout: str | None) -> None:
        axes[f"{name}.weight"] = (pout, pin, None, None)

    def bn(name: str, group: str) -> None:
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            axes[f"{name}.{suffix}"] = (group,)
        axes[f"{name}.num_batches_tracked"] = ()

    conv("conv1", None, "P_0")
    bn("bn1", "P_0")
    stage_in = "P_0"
    for stage in range(1, 5):
        stage_out = "P_0" if stage == 1 else f"P_{stage - 1}"
        for block in range(2):
            prefix = f"layer{stage}.{block}"
            pin = stage_in if block == 0 else stage_out
            inner = f"P_{prefix}_inner"
            conv(f"{prefix}.conv1", pin, inner)
            bn(f"{prefix}.bn1", inner)
            conv(f"{prefix}.conv2", inner, stage_out)
            bn(f"{prefix}.bn2", stage_out)
            if stage > 1 and block == 0:
                conv(f"{prefix}.shortcut.0", pin, stage_out)
                bn(f"{prefix}.shortcut.1", stage_out)
        stage_in = stage_out
    axes["fc.weight"] = (None, "P_3")
    axes["fc.bias"] = (None,)
    return axes


def sample_resnet_gauge(model: nn.Module, seed: int) -> dict[str, torch.Tensor]:
    axes = resnet18slim_permutation_axes()
    group_sizes: dict[str, int] = {}
    state = model.state_dict()
    for key, key_axes in axes.items():
        tensor = state[key]
        for axis, group in enumerate(key_axes):
            if group is None:
                continue
            size = int(tensor.shape[axis])
            old = group_sizes.setdefault(group, size)
            if old != size:
                raise ValueError(f"Permutation group {group} has inconsistent sizes {old} and {size}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return {group: torch.randperm(size, generator=generator) for group, size in sorted(group_sizes.items())}


def apply_resnet_gauge(state: Mapping[str, torch.Tensor], gauge: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    axes = resnet18slim_permutation_axes()
    missing = sorted(set(state) - set(axes))
    if missing:
        raise ValueError(f"Permutation spec does not cover state keys: {missing}")
    output: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        transformed = tensor.detach().clone()
        for axis, group in enumerate(axes[key]):
            if group is not None:
                transformed = transformed.index_select(axis, gauge[group].to(transformed.device))
        output[key] = transformed.contiguous()
    return output


def invert_resnet_gauge(gauge: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the exact inverse of a new-index-to-old-index channel gauge."""

    return {group: torch.argsort(indices) for group, indices in gauge.items()}


def gauge_id(gauge: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(gauge):
        digest.update(key.encode("utf-8"))
        digest.update(gauge[key].to(torch.int64).cpu().numpy().tobytes())
    return f"gauge:{digest.hexdigest()}"


def input_feature_permutation(spec: OperatorSpec, gauge: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return new-to-old column indices for a captured operator context."""

    axes = resnet18slim_permutation_axes()[spec.key]
    input_group = axes[1]
    if input_group is None:
        return torch.arange(spec.matrix_shape[0])
    channel_perm = gauge[input_group].to(torch.long).cpu()
    if spec.operation == "linear":
        return channel_perm
    kh, kw = spec.kernel
    offsets = torch.arange(kh * kw, dtype=torch.long)
    return (channel_perm[:, None] * (kh * kw) + offsets[None, :]).reshape(-1)


def output_feature_permutation(spec: OperatorSpec, gauge: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return new-to-old output-column indices for an operator matrix."""

    axes = resnet18slim_permutation_axes()[spec.key]
    output_group = axes[0]
    if output_group is None:
        return torch.arange(spec.matrix_shape[1])
    return gauge[output_group].to(torch.long).cpu()
