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



def _identity_sample_collate(sample: Any) -> Any:
    return sample



def _sample_list_collate(samples: list[Any]) -> list[Any]:
    return samples



def _offline_loader_worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)



def _flatten_loader_batches(loader_iter: Iterator[Any]) -> Iterator[Any]:
    for item in loader_iter:
        if isinstance(item, list):
            for sample in item:
                yield sample
        else:
            yield item



@dataclass(slots=True)
class SourceSampleRecord:
    x: torch.Tensor
    W: torch.Tensor
    model_name: str = ""
    layer_name: str = ""
    model_run_id: int | None = None



@dataclass(slots=True)
class SourceSliceState:
    source: SourceSampleRecord
    patch_size: int
    row_patch_groups: tuple[torch.Tensor, ...] | None
    col_groups: tuple[torch.Tensor, ...] | None
    row_cursor: int = 0
    col_cursor: int = 0
    single_unsliced_consumed: bool = False



@dataclass(slots=True)
class ConsumedSourceBatch:
    W: torch.Tensor
    x: torch.Tensor
    x_mask: torch.Tensor
    d_in_mask: torch.Tensor
    d_out_mask: torch.Tensor
    used_source_indices: tuple[int, ...]
    next_start_offset: int
    source_pool_size: int
    source_pool_unique_named_models: int
    source_pool_missing_model_names: int
    source_pool_remaining_slices_pre: int
    source_pool_remaining_slices_post: int



@dataclass(slots=True)
class PreparedTrainingBatch:
    W: torch.Tensor
    x: torch.Tensor
    x_mask: torch.Tensor
    d_in_mask: torch.Tensor
    d_out_mask: torch.Tensor
    source_diversity: dict[str, float]
    build_time_s: float
    # Kept separate from producer/build time so the bounded runtime profiler
    # can account for the complete CPU producer path without conflating the
    # synchronous pinning copy with mmap/reconstruction/batching work.
    pin_time_s: float = 0.0
    # Exact logical tile identities actually contained in this batch.  These
    # remain attached while producer prefetch advances beyond consumption.
    logical_indices: tuple[int, ...] = ()



@dataclass(slots=True)
class PreslicedSliceRecord:
    x: torch.Tensor
    W: torch.Tensor
    x_mask: torch.Tensor
    d_in_mask: torch.Tensor
    d_out_mask: torch.Tensor
    model_name: str = ""
    layer_name: str = ""
    logical_index: int | None = None


__all__ = [
    '_identity_sample_collate',
    '_sample_list_collate',
    '_offline_loader_worker_init_fn',
    '_flatten_loader_batches',
    'SourceSampleRecord',
    'SourceSliceState',
    'ConsumedSourceBatch',
    'PreparedTrainingBatch',
    'PreslicedSliceRecord',
]
