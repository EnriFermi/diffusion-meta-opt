"""Deterministic native-train activation capture for operator matrices."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from big_vae.weightclip_benchmark.parameter_adapters import OperatorSpec


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def conv_im2col_rows(inputs: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
    """Rows whose right-multiplication by our matrix equals Conv2d."""

    if module.groups != 1:
        raise ValueError("primary im2col adapter only supports groups=1")
    unfolded = F.unfold(
        inputs,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])


def operator_rows(inputs: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Conv2d):
        return conv_im2col_rows(inputs, module)
    if isinstance(module, nn.Linear):
        return inputs.reshape(-1, inputs.shape[-1])
    raise TypeError(f"Unsupported operator module: {type(module).__name__}")


class UniformStreamingReservoir:
    """Exact uniform fixed-size sample without retaining all activation rows."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.generator = np.random.default_rng(int(seed))
        self.rows: torch.Tensor | None = None
        self.seen = 0

    def update(self, rows: torch.Tensor) -> None:
        if rows.ndim != 2:
            raise ValueError(f"rows must be rank-2, got {tuple(rows.shape)}")
        count = int(rows.shape[0])
        if count == 0:
            return
        if not torch.isfinite(rows).all():
            raise ValueError("activation rows contain NaN/Inf")
        rows = rows.detach()
        old_seen = self.seen
        self.seen += count
        if self.rows is None:
            take = min(self.capacity, count)
            idx = self.generator.choice(count, size=take, replace=False)
            self.rows = rows[torch.as_tensor(idx, device=rows.device)].to("cpu", torch.float32).contiguous()
            return

        target = min(self.capacity, self.seen)
        # A target-size uniform sample from old_seen + count has a
        # hypergeometric number of rows from the new chunk.  The existing
        # reservoir is already uniform conditional on selecting an old row.
        new_take = int(self.generator.hypergeometric(count, old_seen, target))
        old_take = target - new_take
        if old_take:
            old_idx = self.generator.choice(len(self.rows), size=old_take, replace=False)
            old_rows = self.rows[torch.as_tensor(old_idx)]
        else:
            old_rows = self.rows[:0]
        if new_take:
            new_idx = self.generator.choice(count, size=new_take, replace=False)
            new_rows = rows[torch.as_tensor(new_idx, device=rows.device)].to("cpu", torch.float32)
        else:
            new_rows = self.rows[:0]
        self.rows = torch.cat((old_rows, new_rows), dim=0).contiguous()

    def result(self) -> torch.Tensor:
        if self.rows is None:
            raise RuntimeError("reservoir received no rows")
        return self.rows


class NativeActivationCapture:
    """Pre-forward hooks collecting operator-domain rows for named modules."""

    def __init__(
        self,
        model: nn.Module,
        specs: Iterable[OperatorSpec],
        *,
        max_rows: int,
        seed_parts: tuple[object, ...],
    ) -> None:
        modules = dict(model.named_modules())
        self._reservoirs: dict[str, UniformStreamingReservoir] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for spec in specs:
            module = modules[spec.module_name]
            reservoir = UniformStreamingReservoir(max_rows, stable_seed(*seed_parts, spec.key))
            self._reservoirs[spec.key] = reservoir

            def hook(current: nn.Module, args: tuple[torch.Tensor, ...], *, target=reservoir) -> None:
                target.update(operator_rows(args[0], current))

            self._handles.append(module.register_forward_pre_hook(hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "NativeActivationCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def results(self) -> dict[str, torch.Tensor]:
        return {key: reservoir.result() for key, reservoir in self._reservoirs.items()}


@torch.inference_mode()
def capture_native_train_activations(
    model: nn.Module,
    train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    specs: Iterable[OperatorSpec],
    *,
    max_rows: int,
    seed_parts: tuple[object, ...],
    device: torch.device | str,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """Capture only from the caller-provided native *training* iterator."""

    model.eval()
    with NativeActivationCapture(model, specs, max_rows=max_rows, seed_parts=seed_parts) as capture:
        for batch_index, (images, _) in enumerate(train_batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            model(images.to(device, non_blocking=True))
    return capture.results()


@dataclass(frozen=True, slots=True)
class ContextTile:
    raw_rows: torch.Tensor
    sample_mask: torch.Tensor
    feature_mask: torch.Tensor
    quantiles: torch.Tensor
    mean: torch.Tensor
    log_scale: torch.Tensor
    covariance: torch.Tensor
    row_start: int
    valid_features: int


def context_tiles(
    rows: torch.Tensor,
    *,
    tile_width: int = 128,
    max_rows: int = 512,
    num_quantiles: int = 16,
) -> list[ContextTile]:
    return context_tiles_batched(
        {"context": rows},
        tile_width=tile_width,
        max_rows=max_rows,
        num_quantiles=num_quantiles,
    )["context"]


def context_tiles_batched(
    rows_by_key: Mapping[str, torch.Tensor],
    *,
    tile_width: int = 128,
    max_rows: int = 512,
    num_quantiles: int = 16,
    compute_device: torch.device | str = "cpu",
) -> dict[str, list[ContextTile]]:
    """Summarize all feature tiles in a few batched kernels.

    Calling ``torch.quantile`` and GEMM once per 128-wide tile made a single
    ResNet checkpoint spend roughly two minutes in Python/kernel-launch
    overhead.  Grouping tiles by sample count preserves each tile's statistics
    while reducing the same real checkpoint to a few seconds.  The chosen
    compute device must be part of the caller's immutable bank contract.
    """

    if not rows_by_key:
        raise ValueError("rows_by_key must be nonempty")
    outputs: dict[str, list[ContextTile]] = {str(key): [] for key in rows_by_key}
    # Metadata and already tiled tensors are grouped by sample count so each
    # statistical kernel is launched once per group, not once per tile.
    descriptors_by_samples: dict[int, list[tuple[str, int, int]]] = {}
    batches_by_samples: dict[int, list[torch.Tensor]] = {}
    for raw_key, source in rows_by_key.items():
        key = str(raw_key)
        if source.ndim != 2 or source.shape[0] > max_rows:
            raise ValueError(f"{key}: expected [n,d] with n <= {max_rows}, got {tuple(source.shape)}")
        rows = source.to(torch.float32).cpu()
        n, width = rows.shape
        if n == 0 or width == 0:
            raise ValueError(f"{key}: context must have positive sample and feature dimensions")
        chunks = math.ceil(width / tile_width)
        padded = torch.zeros((n, chunks * tile_width), dtype=torch.float32)
        padded[:, :width] = rows
        tiled = padded.reshape(n, chunks, tile_width).permute(1, 0, 2).contiguous()
        batches_by_samples.setdefault(n, []).append(tiled)
        for chunk in range(chunks):
            start = chunk * tile_width
            valid = min(tile_width, width - start)
            descriptors_by_samples.setdefault(n, []).append((key, start, valid))

    stats_device = torch.device(compute_device)
    if stats_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA context-summary device is unavailable: {stats_device}")
    probs = torch.linspace(0.0, 1.0, num_quantiles, device=stats_device)
    for n, descriptors in descriptors_by_samples.items():
        values = torch.cat(batches_by_samples[n], dim=0)
        if not torch.isfinite(values).all():
            raise ValueError("context contains NaN/Inf")
        stats_values = values.to(stats_device, non_blocking=stats_device.type == "cuda")
        quantiles = torch.quantile(stats_values, probs, dim=1).permute(1, 0, 2).contiguous().cpu()
        means_device = stats_values.mean(dim=1)
        scales_device = stats_values.std(dim=1, unbiased=False)
        centered = stats_values - means_device[:, None, :]
        covariances = (
            torch.bmm(centered.transpose(1, 2), centered) / max(1, n - 1)
        ).cpu()
        means = means_device.cpu()
        scales = scales_device.cpu()
        raw_batch = torch.zeros((len(descriptors), max_rows, tile_width), dtype=torch.float32)
        raw_batch[:, :n] = values
        sample_mask = torch.zeros(max_rows, dtype=torch.bool)
        sample_mask[:n] = True
        feature_masks = torch.arange(tile_width)[None, :] < torch.tensor(
            [descriptor[2] for descriptor in descriptors]
        )[:, None]
        for index, (key, start, valid) in enumerate(descriptors):
            outputs[key].append(
                ContextTile(
                    raw_rows=raw_batch[index],
                    sample_mask=sample_mask,
                    feature_mask=feature_masks[index],
                    quantiles=quantiles[index],
                    mean=means[index],
                    log_scale=torch.log(scales[index] + 1e-6),
                    covariance=covariances[index],
                    row_start=start,
                    valid_features=valid,
                )
            )
    return outputs


def verify_conv_operator(module: nn.Conv2d, inputs: torch.Tensor, matrix: torch.Tensor) -> tuple[float, float]:
    """Numerically compare Conv2d with unfold(rows) @ operator matrix."""

    direct = module(inputs)
    unfolded = F.unfold(inputs, module.kernel_size, module.dilation, module.padding, module.stride)
    reconstructed = unfolded.transpose(1, 2) @ matrix
    reconstructed = reconstructed.transpose(1, 2).reshape_as(direct)
    diff = (direct - reconstructed).abs()
    return float(diff.max().item()), float(diff.mean().item())
