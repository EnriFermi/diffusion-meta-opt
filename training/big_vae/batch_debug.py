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
from dataset.big_vae_offline import (
    ensure_presliced_big_vae_dataset,
    offline_big_vae_data_pipeline,
    presliced_big_vae_data_pipeline,
)
from dataset.logging_utils import LOG_PATH_ENV, configure_process_logging, resolve_process_log_path
from experiments.background_prefetch import BackgroundPrefetcher
from models.weight_quantile_vae import (
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

from training.big_vae.data_types import PreparedTrainingBatch



def _compute_curriculum_slice_sizes(cfg: DictConfig) -> tuple[int, int]:
    train_cfg = cfg.get("train", {})
    stage = max(1, int(train_cfg.get("stage", 1)))
    base_T = int(train_cfg.get("stage_base_T_patches", 4))
    base_d_out = int(train_cfg.get("stage_base_d_out", 16))
    scale = int(train_cfg.get("stage_scale_factor", 2))
    max_T = base_T * (scale ** (stage - 1))
    max_d_out = base_d_out * (scale ** (stage - 1))
    return max_T, max_d_out



def _stable_batch_shape_targets(
    *,
    cfg: DictConfig,
    patch_size: int,
    max_T_patches: int,
    max_d_out: int,
    max_x_rows: int,
) -> tuple[int | None, int | None, int | None]:
    if not bool(cfg.train.get("compile_stable_batch_shapes", False)):
        return None, None, None
    target_x_rows = int(max_x_rows) if int(max_x_rows) > 0 else None
    target_d_in = int(patch_size) * int(max_T_patches)
    target_d_out = int(max_d_out)
    return target_x_rows, target_d_in, target_d_out



def _tensor_debug_stats(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"is_none": True}

    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = int(flat.numel())
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    payload: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    if finite_count > 0:
        finite_values = flat[finite_mask]
        payload["min"] = float(finite_values.min().item())
        payload["max"] = float(finite_values.max().item())
        payload["abs_max"] = float(finite_values.abs().max().item())
        payload["mean"] = float(finite_values.mean().item())
        payload["std"] = float(finite_values.std(unbiased=False).item())
    return payload



def _pin_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor



def _pin_prepared_training_batch(batch: PreparedTrainingBatch) -> PreparedTrainingBatch:
    return PreparedTrainingBatch(
        W=_pin_tensor_if_available(batch.W),
        x=_pin_tensor_if_available(batch.x),
        x_mask=_pin_tensor_if_available(batch.x_mask),
        d_in_mask=_pin_tensor_if_available(batch.d_in_mask),
        d_out_mask=_pin_tensor_if_available(batch.d_out_mask),
        source_diversity=dict(batch.source_diversity),
        build_time_s=float(batch.build_time_s),
    )



def _scalar_debug_value(tensor: torch.Tensor) -> float | str:
    value = tensor.detach()
    if value.numel() != 1:
        return f"<non_scalar shape={tuple(value.shape)}>"
    scalar = value.item()
    return float(scalar) if isinstance(scalar, (int, float)) else str(scalar)


__all__ = [
    '_compute_curriculum_slice_sizes',
    '_stable_batch_shape_targets',
    '_tensor_debug_stats',
    '_pin_tensor_if_available',
    '_pin_prepared_training_batch',
    '_scalar_debug_value',
]
