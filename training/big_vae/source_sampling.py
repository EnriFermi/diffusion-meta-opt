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

from training.big_vae.data_types import SourceSampleRecord



def _source_sample_record_to_device(record: SourceSampleRecord, device: torch.device) -> SourceSampleRecord:
    return SourceSampleRecord(
        x=record.x.to(device=device, non_blocking=True),
        W=record.W.to(device=device, non_blocking=True),
        model_name=record.model_name,
        layer_name=record.layer_name,
        model_run_id=record.model_run_id,
    )



def _normalize_batch_source_uniqueness(value: Any) -> str:
    normalized = str(value or "none").strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "model": "model",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"none", "model"}:
        raise ValueError(
            "train.batch_source_mixing.uniqueness must be one of {'none', 'model'}, "
            f"got {value!r}"
        )
    return resolved



def _source_sample_uniqueness_key(record: SourceSampleRecord, uniqueness: str) -> str | None:
    if uniqueness == "none":
        return None
    if uniqueness == "model":
        key = str(record.model_name).strip()
        return key or None
    raise ValueError(f"Unsupported source uniqueness mode: {uniqueness!r}")



def _next_valid_source_sample(
    dataset_iter: Iterator[Any],
    max_x_rows: int,
    logger: logging.Logger,
) -> SourceSampleRecord:
    attempts = 0
    while True:
        sample = next(dataset_iter)
        x = sample.x
        W = sample.weight

        valid = (
            torch.is_tensor(x)
            and torch.is_tensor(W)
            and x.ndim == 2
            and W.ndim == 2
            and x.shape[1] == W.shape[0]
            and x.shape[0] > 0
            and W.shape[0] > 0
            and W.shape[1] > 0
        )
        if not valid:
            attempts += 1
            if attempts % 100 == 0:
                logger.warning("Skipping invalid sample repeatedly; attempts=%s", attempts)
            continue

        x = _prepare_cpu_sample_tensor(x)
        W = _prepare_cpu_sample_tensor(W)

        if max_x_rows > 0 and x.shape[0] > max_x_rows:
            keep = torch.randperm(x.shape[0])[:max_x_rows]
            x = x[keep]

        model_name = str(getattr(sample, "model_name", "")).strip()
        layer_name = str(getattr(sample, "layer_name", "")).strip()
        model_run_id: int | None = None
        sample_meta = getattr(sample, "meta", None)
        if isinstance(sample_meta, dict):
            raw_run_id = sample_meta.get("model_run_id")
            if raw_run_id is not None:
                try:
                    model_run_id = int(raw_run_id)
                except (TypeError, ValueError):
                    model_run_id = None

        return SourceSampleRecord(
            x=x,
            W=W,
            model_name=model_name,
            layer_name=layer_name,
            model_run_id=model_run_id,
        )



def _prepare_cpu_sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        return tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor



def _broadcast_tensor_2d(
    tensor: torch.Tensor | None,
    device: torch.device,
    src: int = 0,
) -> torch.Tensor:
    if not dist.is_initialized():
        if tensor is None:
            raise ValueError("tensor cannot be None when distributed is not initialized")
        return tensor.to(device=device, non_blocking=True)

    rank = dist.get_rank()
    shape = torch.zeros(2, dtype=torch.long, device=device)
    if rank == src:
        if tensor is None:
            raise ValueError("Source rank must provide tensor")
        if tensor.ndim != 2:
            raise ValueError(f"Expected rank-2 tensor, got shape {tuple(tensor.shape)}")
        shape[0] = tensor.shape[0]
        shape[1] = tensor.shape[1]

    dist.broadcast(shape, src=src)

    if rank == src:
        payload = tensor.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    else:
        payload = torch.empty((int(shape[0].item()), int(shape[1].item())), device=device, dtype=torch.float32)

    dist.broadcast(payload, src=src)
    return payload



def _fetch_batch_cpu(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    record = _fetch_source_sample_record_cpu(
        rank=rank,
        device=device,
        dataset_iter=dataset_iter,
        use_broadcast=use_broadcast,
        max_x_rows=max_x_rows,
        logger=logger,
    )
    return record.x, record.W



def _fetch_source_sample_record_cpu(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> SourceSampleRecord:
    if use_broadcast:
        source_record: SourceSampleRecord | None = None
        x_cpu: torch.Tensor | None = None
        W_cpu: torch.Tensor | None = None
        if rank == 0:
            if dataset_iter is None:
                raise RuntimeError("rank0 requires dataset iterator in broadcast mode")
            source_record = _next_valid_source_sample(
                dataset_iter=dataset_iter,
                max_x_rows=max_x_rows,
                logger=logger,
            )
            x_cpu = source_record.x
            W_cpu = source_record.W

        x = _broadcast_tensor_2d(x_cpu, device=device, src=0)
        W = _broadcast_tensor_2d(W_cpu, device=device, src=0)
        return SourceSampleRecord(
            x=_prepare_cpu_sample_tensor(x),
            W=_prepare_cpu_sample_tensor(W),
            model_name=source_record.model_name if source_record is not None else "",
            layer_name=source_record.layer_name if source_record is not None else "",
            model_run_id=source_record.model_run_id if source_record is not None else None,
        )

    if dataset_iter is None:
        raise RuntimeError("dataset iterator is required for sharded mode")

    return _next_valid_source_sample(
        dataset_iter=dataset_iter,
        max_x_rows=max_x_rows,
        logger=logger,
    )



def _fetch_batch(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_cpu, W_cpu = _fetch_batch_cpu(
        rank=rank,
        device=device,
        dataset_iter=dataset_iter,
        use_broadcast=use_broadcast,
        max_x_rows=max_x_rows,
        logger=logger,
    )
    return (
        x_cpu.to(device=device, non_blocking=True),
        W_cpu.to(device=device, non_blocking=True),
    )



def _fetch_source_samples(
    *,
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
    num_samples: int,
    uniqueness: str,
    deferred_samples: deque[SourceSampleRecord] | None = None,
    max_deferred_samples: int = 0,
    min_d_in: int = 0,
    min_d_out: int = 0,
    existing_uniqueness_keys: set[str] | None = None,
) -> list[SourceSampleRecord]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if use_broadcast and uniqueness != "none":
        raise ValueError("batch_source_mixing.uniqueness != 'none' is not supported with distributed broadcast mode")

    selected: list[SourceSampleRecord] = []
    fallback_sample: SourceSampleRecord | None = None
    fallback_shape_compatible_sample: SourceSampleRecord | None = None
    uniqueness_keys: set[str] = set(existing_uniqueness_keys or ())
    attempts = 0
    max_attempts = max(int(num_samples) * 8, int(num_samples))
    if min_d_in > 0 or min_d_out > 0:
        max_attempts = max(max_attempts, 128)

    def _is_shape_compatible(record: SourceSampleRecord) -> bool:
        if min_d_in > 0 and int(record.W.shape[0]) < int(min_d_in):
            return False
        if min_d_out > 0 and int(record.W.shape[1]) < int(min_d_out):
            return False
        return True

    def _try_select(record: SourceSampleRecord) -> bool:
        if not _is_shape_compatible(record):
            return False
        uniqueness_key = _source_sample_uniqueness_key(record, uniqueness)
        if uniqueness_key is not None and uniqueness_key in uniqueness_keys:
            return False
        selected.append(record)
        if uniqueness_key is not None:
            uniqueness_keys.add(uniqueness_key)
        return True

    if deferred_samples is not None and deferred_samples:
        deferred_remaining: deque[SourceSampleRecord] = deque()
        while deferred_samples and len(selected) < num_samples:
            candidate = deferred_samples.popleft()
            if not _try_select(candidate):
                deferred_remaining.append(candidate)
        while deferred_samples:
            deferred_remaining.append(deferred_samples.popleft())
        deferred_samples.extend(deferred_remaining)

    while len(selected) < num_samples and attempts < max_attempts:
        record = _fetch_source_sample_record_cpu(
            rank=rank,
            device=device,
            dataset_iter=dataset_iter,
            use_broadcast=use_broadcast,
            max_x_rows=max_x_rows,
            logger=logger,
        )
        attempts += 1

        if fallback_sample is None:
            fallback_sample = record
        if fallback_shape_compatible_sample is None and _is_shape_compatible(record):
            fallback_shape_compatible_sample = record

        if _try_select(record):
            continue
        if deferred_samples is not None and max_deferred_samples > 0 and _is_shape_compatible(record):
            if len(deferred_samples) >= max_deferred_samples:
                deferred_samples.popleft()
            deferred_samples.append(record)

    if not selected:
        if fallback_shape_compatible_sample is not None:
            selected.append(fallback_shape_compatible_sample)
        elif fallback_sample is None:
            raise RuntimeError("failed to fetch any source samples")
        elif min_d_in <= 0 and min_d_out <= 0:
            selected.append(fallback_sample)
        else:
            logger.warning(
                "Failed to fetch shape-compatible source samples after %s attempts; min_d_in=%s min_d_out=%s",
                attempts,
                min_d_in,
                min_d_out,
            )
            return []

    return selected


__all__ = [
    '_source_sample_record_to_device',
    '_normalize_batch_source_uniqueness',
    '_source_sample_uniqueness_key',
    '_next_valid_source_sample',
    '_prepare_cpu_sample_tensor',
    '_broadcast_tensor_2d',
    '_fetch_batch_cpu',
    '_fetch_source_sample_record_cpu',
    '_fetch_batch',
    '_fetch_source_samples',
]
