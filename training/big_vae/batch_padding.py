from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import data_pipeline, setup_logging
from big_vae.datasets.offline import (
    ensure_presliced_big_vae_dataset,
    offline_big_vae_data_pipeline,
    presliced_big_vae_data_pipeline,
)
from dataset.logging_utils import LOG_PATH_ENV, configure_process_logging, resolve_process_log_path
from experiments.background_prefetch import BackgroundPrefetcher
from big_vae.models import (
    BigVAEConfig,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
    WeightQuantileVAE,
    build_weight_quantile_vae,
)
from training.optim import build_adamw_optimizer, build_cosine_scheduler
from training.forensics import (
    apply_nccl_forensics_env,
    emit_fatal_report,
    maybe_enable_core_dumps,
    maybe_redirect_stdio,
    monitor_send_event,
    start_process_monitor,
    stop_process_monitor,
)
from training.runtime import (
    autocast_context as runtime_autocast_context,
    configure_per_run_artifacts as runtime_configure_per_run_artifacts,
    create_grad_scaler as runtime_create_grad_scaler,
    find_free_port as runtime_find_free_port,
    get_rank_logger,
    maybe_compile_model,
    resolve_amp as runtime_resolve_amp,
    resolve_backend as runtime_resolve_backend,
    resolve_device as runtime_resolve_device,
    resolve_world_size as runtime_resolve_world_size,
    seed_everything as runtime_seed_everything,
    set_speed_optimizations as runtime_set_speed_optimizations,
)

from training.big_vae.data_types import SourceSampleRecord



def _sample_synthetic_layer(
    *,
    device: torch.device,
    n_rows: int,
    d_in: int,
    d_out: int,
    x_std: float,
    w_std: float,
    max_x_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_n_rows = int(n_rows)
    if max_x_rows > 0:
        effective_n_rows = min(effective_n_rows, int(max_x_rows))
    x = torch.randn((effective_n_rows, int(d_in)), device=device, dtype=torch.float32) * float(x_std)
    W = torch.randn((int(d_in), int(d_out)), device=device, dtype=torch.float32) * float(w_std)
    return x, W



def _round_robin_source_indices(num_sources: int, batch_size: int, start_offset: int = 0) -> list[int]:
    if num_sources <= 0:
        raise ValueError(f"num_sources must be > 0, got {num_sources}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    start = int(start_offset) % int(num_sources)
    return [int((start + batch_idx) % num_sources) for batch_idx in range(batch_size)]



def _compute_batch_source_diversity_stats(
    source_samples: Sequence[SourceSampleRecord] | None,
    *,
    batch_size: int,
    start_offset: int = 0,
) -> dict[str, float]:
    stats: dict[str, float] = {
        "source_pool_size": 0.0,
        "source_pool_unique_named_models": 0.0,
        "source_pool_missing_model_names": 0.0,
        "batch_sources_used": 0.0,
        "batch_unique_models": 0.0,
        "batch_unique_named_models": 0.0,
        "batch_missing_model_sources": 0.0,
        "batch_model_entropy": 0.0,
        "batch_model_perplexity": 0.0,
    }
    if not source_samples:
        return stats

    pool_named_models = {
        str(record.model_name).strip()
        for record in source_samples
        if str(record.model_name).strip()
    }
    pool_missing_model_names = sum(
        1
        for record in source_samples
        if not str(record.model_name).strip()
    )
    stats["source_pool_size"] = float(len(source_samples))
    stats["source_pool_unique_named_models"] = float(len(pool_named_models))
    stats["source_pool_missing_model_names"] = float(pool_missing_model_names)

    assignment = _round_robin_source_indices(
        num_sources=len(source_samples),
        batch_size=batch_size,
        start_offset=start_offset,
    )
    source_counts = [0] * len(source_samples)
    for source_idx in assignment:
        source_counts[source_idx] += 1

    used_indices = [source_idx for source_idx, count in enumerate(source_counts) if count > 0]
    batch_named_models: set[str] = set()
    batch_missing_model_sources = 0
    effective_model_counts: dict[str, int] = {}
    for source_idx in used_indices:
        model_name = str(source_samples[source_idx].model_name).strip()
        if model_name:
            batch_named_models.add(model_name)
            label = model_name
        else:
            batch_missing_model_sources += 1
            label = f"__unknown_source_{source_idx}"
        effective_model_counts[label] = effective_model_counts.get(label, 0) + int(source_counts[source_idx])

    entropy = 0.0
    total_items = sum(effective_model_counts.values())
    if total_items > 0:
        for count in effective_model_counts.values():
            prob = float(count) / float(total_items)
            entropy -= prob * math.log(max(prob, 1e-12))

    stats["batch_sources_used"] = float(len(used_indices))
    stats["batch_unique_models"] = float(len(effective_model_counts))
    stats["batch_unique_named_models"] = float(len(batch_named_models))
    stats["batch_missing_model_sources"] = float(batch_missing_model_sources)
    stats["batch_model_entropy"] = float(entropy)
    stats["batch_model_perplexity"] = float(math.exp(entropy)) if total_items > 0 else 0.0
    return stats



def _pad_x_rows_with_mask(x: torch.Tensor, target_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if target_rows <= 0:
        raise ValueError(f"target_rows must be > 0, got {target_rows}")
    current_rows = int(x.shape[0])
    if current_rows > int(target_rows):
        raise ValueError(f"target_rows ({target_rows}) must be >= current_rows ({current_rows})")

    valid_mask = torch.ones((current_rows,), device=x.device, dtype=torch.bool)
    if current_rows == int(target_rows):
        return x, valid_mask

    pad_rows = int(target_rows) - current_rows
    x_pad = torch.zeros((pad_rows, int(x.shape[1])), device=x.device, dtype=x.dtype)
    mask_pad = torch.zeros((pad_rows,), device=x.device, dtype=torch.bool)
    return torch.cat([x, x_pad], dim=0), torch.cat([valid_mask, mask_pad], dim=0)



def _pad_d_in_with_mask(
    x: torch.Tensor,
    W: torch.Tensor,
    target_d_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_d_in <= 0:
        raise ValueError(f"target_d_in must be > 0, got {target_d_in}")
    if x.ndim not in {2, 3}:
        raise ValueError(f"x must be rank-2 or rank-3, got {tuple(x.shape)}")
    if W.ndim not in {2, 3}:
        raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
    current_d_in = int(W.shape[-2])
    if int(x.shape[-1]) != current_d_in:
        raise ValueError(f"x last dim ({int(x.shape[-1])}) must match W d_in ({current_d_in})")
    if current_d_in > int(target_d_in):
        raise ValueError(f"target_d_in ({target_d_in}) must be >= current_d_in ({current_d_in})")

    mask_shape = (int(target_d_in),) if W.ndim == 2 else (int(W.shape[0]), int(target_d_in))
    d_in_mask = torch.zeros(mask_shape, device=W.device, dtype=torch.bool)
    d_in_mask[..., :current_d_in] = True
    if current_d_in == int(target_d_in):
        return x, W, d_in_mask

    pad_d_in = int(target_d_in) - current_d_in
    if x.ndim == 2:
        x_pad = torch.zeros((int(x.shape[0]), pad_d_in), device=x.device, dtype=x.dtype)
        W_pad = torch.zeros((pad_d_in, int(W.shape[1])), device=W.device, dtype=W.dtype)
        return torch.cat([x, x_pad], dim=1), torch.cat([W, W_pad], dim=0), d_in_mask

    x_pad = torch.zeros((int(x.shape[0]), int(x.shape[1]), pad_d_in), device=x.device, dtype=x.dtype)
    W_pad = torch.zeros((int(W.shape[0]), pad_d_in, int(W.shape[2])), device=W.device, dtype=W.dtype)
    return torch.cat([x, x_pad], dim=2), torch.cat([W, W_pad], dim=1), d_in_mask



def _pad_d_out_with_mask(
    W: torch.Tensor,
    target_d_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_d_out <= 0:
        raise ValueError(f"target_d_out must be > 0, got {target_d_out}")
    if W.ndim not in {2, 3}:
        raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")

    current_d_out = int(W.shape[-1])
    if current_d_out > int(target_d_out):
        raise ValueError(f"target_d_out ({target_d_out}) must be >= current_d_out ({current_d_out})")

    mask_shape = (int(target_d_out),) if W.ndim == 2 else (int(W.shape[0]), int(target_d_out))
    d_out_mask = torch.zeros(mask_shape, device=W.device, dtype=torch.bool)
    d_out_mask[..., :current_d_out] = True
    if current_d_out == int(target_d_out):
        return W, d_out_mask

    pad_d_out = int(target_d_out) - current_d_out
    if W.ndim == 2:
        W_pad = torch.zeros((int(W.shape[0]), pad_d_out), device=W.device, dtype=W.dtype)
        return torch.cat([W, W_pad], dim=1), d_out_mask

    W_pad = torch.zeros((int(W.shape[0]), int(W.shape[1]), pad_d_out), device=W.device, dtype=W.dtype)
    return torch.cat([W, W_pad], dim=2), d_out_mask



def _materialize_padded_slice_batch(
    raw_slices: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    target_x_rows: int,
    target_d_in: int,
    target_d_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not raw_slices:
        raise ValueError("raw_slices must not be empty")
    batch_size = int(len(raw_slices))
    first_W, first_x = raw_slices[0]
    W_batch = first_W.new_zeros((batch_size, int(target_d_in), int(target_d_out)))
    x_batch = first_x.new_zeros((batch_size, int(target_x_rows), int(target_d_in)))
    x_mask_batch = torch.zeros((batch_size, int(target_x_rows)), device=first_x.device, dtype=torch.bool)
    d_in_mask_batch = torch.zeros((batch_size, int(target_d_in)), device=first_W.device, dtype=torch.bool)
    d_out_mask_batch = torch.zeros((batch_size, int(target_d_out)), device=first_W.device, dtype=torch.bool)

    for batch_idx, (W_i, x_i) in enumerate(raw_slices):
        x_rows = int(x_i.shape[0])
        d_in_i, d_out_i = map(int, W_i.shape)
        if x_rows > int(target_x_rows):
            raise ValueError(f"target_x_rows ({target_x_rows}) must be >= current_rows ({x_rows})")
        if d_in_i > int(target_d_in):
            raise ValueError(f"target_d_in ({target_d_in}) must be >= current_d_in ({d_in_i})")
        if d_out_i > int(target_d_out):
            raise ValueError(f"target_d_out ({target_d_out}) must be >= current_d_out ({d_out_i})")
        W_batch[batch_idx, :d_in_i, :d_out_i] = W_i
        x_batch[batch_idx, :x_rows, :d_in_i] = x_i
        x_mask_batch[batch_idx, :x_rows] = True
        d_in_mask_batch[batch_idx, :d_in_i] = True
        d_out_mask_batch[batch_idx, :d_out_i] = True

    return W_batch, x_batch, x_mask_batch, d_in_mask_batch, d_out_mask_batch



def _slice_sample(
    W: torch.Tensor,
    x: torch.Tensor,
    max_T_patches: int,
    max_d_out: int,
    patch_size: int,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_in, d_out = W.shape
    T_total = d_in // patch_size
    T_use = min(max_T_patches, T_total)
    d_out_use = min(max_d_out, d_out)

    need_row_slice = T_use < T_total
    need_col_slice = d_out_use < d_out
    dev = W.device

    if not need_row_slice and not need_col_slice:
        return W.unsqueeze(0).expand(batch_size, -1, -1), x.unsqueeze(0).expand(batch_size, -1, -1)

    n_rows = int(x.shape[0])
    if need_row_slice:
        offsets = torch.arange(patch_size, device=dev, dtype=torch.long)
        row_scores = torch.rand((int(batch_size), int(T_total)), device=dev, dtype=torch.float32)
        patch_idx = row_scores.topk(k=int(T_use), dim=1, largest=False).indices.sort(dim=1).values
        row_idx = (patch_idx.unsqueeze(-1) * int(patch_size) + offsets.view(1, 1, -1)).reshape(int(batch_size), -1)
        row_idx = row_idx.clamp(max=int(d_in) - 1)
        W_batch = W.unsqueeze(0).expand(int(batch_size), -1, -1).gather(
            1,
            row_idx.unsqueeze(-1).expand(-1, -1, int(d_out)),
        )
        x_batch = x.unsqueeze(0).expand(int(batch_size), -1, -1).gather(
            2,
            row_idx.unsqueeze(1).expand(-1, n_rows, -1),
        )
    else:
        W_batch = W.unsqueeze(0).expand(int(batch_size), -1, -1)
        x_batch = x.unsqueeze(0).expand(int(batch_size), -1, -1)

    if need_col_slice:
        col_scores = torch.rand((int(batch_size), int(d_out)), device=dev, dtype=torch.float32)
        col_idx = col_scores.topk(k=int(d_out_use), dim=1, largest=False).indices.sort(dim=1).values
        W_batch = W_batch.gather(
            2,
            col_idx.unsqueeze(1).expand(-1, int(W_batch.shape[1]), -1),
        )

    return W_batch, x_batch


__all__ = [
    '_sample_synthetic_layer',
    '_round_robin_source_indices',
    '_compute_batch_source_diversity_stats',
    '_pad_x_rows_with_mask',
    '_pad_d_in_with_mask',
    '_pad_d_out_with_mask',
    '_materialize_padded_slice_batch',
    '_slice_sample',
]
