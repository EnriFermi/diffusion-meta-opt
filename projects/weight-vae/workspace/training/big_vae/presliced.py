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
from dataset.big_vae_offline_parts.metadata import _prepare_cpu_sample_tensor
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

from training.big_vae.data_types import PreparedTrainingBatch, PreslicedSliceRecord



def _prepare_cpu_bool_mask(
    tensor: Any,
    *,
    target_shape: tuple[int, ...],
    default_true: bool,
) -> torch.Tensor:
    if torch.is_tensor(tensor):
        mask = tensor.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
        if tuple(mask.shape) != tuple(target_shape):
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match expected {tuple(target_shape)}")
        return mask
    fill_value = bool(default_true)
    return torch.full(tuple(int(dim) for dim in target_shape), fill_value, dtype=torch.bool)



def _next_valid_presliced_slice(
    dataset_iter: Iterator[Any],
    logger: logging.Logger,
) -> PreslicedSliceRecord:
    attempts = 0
    while True:
        sample = next(dataset_iter)
        x = getattr(sample, "x", None)
        W = getattr(sample, "weight", None)
        meta = getattr(sample, "meta", {}) or {}
        try:
            valid = (
                torch.is_tensor(x)
                and torch.is_tensor(W)
                and x.ndim == 2
                and W.ndim == 2
                and int(x.shape[1]) == int(W.shape[0])
                and int(x.shape[0]) > 0
                and int(W.shape[0]) > 0
                and int(W.shape[1]) > 0
            )
            if not valid:
                raise ValueError("invalid presliced W/x tensor shapes")

            x_cpu = _prepare_cpu_sample_tensor(x)
            W_cpu = _prepare_cpu_sample_tensor(W)
            if not isinstance(meta, dict):
                meta = {}
            x_mask = _prepare_cpu_bool_mask(
                meta.get("x_mask"),
                target_shape=(int(x_cpu.shape[0]),),
                default_true=True,
            )
            d_in_mask = _prepare_cpu_bool_mask(
                meta.get("d_in_mask"),
                target_shape=(int(W_cpu.shape[0]),),
                default_true=True,
            )
            d_out_mask = _prepare_cpu_bool_mask(
                meta.get("d_out_mask"),
                target_shape=(int(W_cpu.shape[1]),),
                default_true=True,
            )
            return PreslicedSliceRecord(
                x=x_cpu,
                W=W_cpu,
                x_mask=x_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                model_name=str(getattr(sample, "model_name", "")).strip(),
                layer_name=str(getattr(sample, "layer_name", "")).strip(),
                logical_index=(
                    int(meta["logical_index"])
                    if isinstance(meta, dict) and meta.get("logical_index") is not None
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            attempts += 1
            if attempts % 100 == 0:
                logger.warning("Skipping invalid presliced samples repeatedly; attempts=%s error=%s", attempts, exc)



def _compute_presliced_batch_source_diversity_stats(
    records: Sequence[PreslicedSliceRecord],
) -> dict[str, float]:
    stats: dict[str, float] = {
        "source_pool_size": float(len(records)),
        "source_pool_unique_named_models": 0.0,
        "source_pool_missing_model_names": 0.0,
        "source_pool_remaining_slices_pre": 0.0,
        "source_pool_remaining_slices_post": 0.0,
        "batch_sources_used": float(len(records)),
        "batch_unique_models": 0.0,
        "batch_unique_named_models": 0.0,
        "batch_missing_model_sources": 0.0,
        "batch_model_entropy": 0.0,
        "batch_model_perplexity": 0.0,
    }
    if not records:
        return stats

    named_models = {str(record.model_name).strip() for record in records if str(record.model_name).strip()}
    missing_model_sources = sum(1 for record in records if not str(record.model_name).strip())
    model_counts: dict[str, int] = {}
    for idx, record in enumerate(records):
        model_name = str(record.model_name).strip()
        label = model_name if model_name else f"__unknown_source_{idx}"
        model_counts[label] = model_counts.get(label, 0) + 1

    entropy = 0.0
    total_items = sum(model_counts.values())
    if total_items > 0:
        for count in model_counts.values():
            prob = float(count) / float(total_items)
            entropy -= prob * math.log(max(prob, 1e-12))

    stats["source_pool_unique_named_models"] = float(len(named_models))
    stats["source_pool_missing_model_names"] = float(missing_model_sources)
    stats["batch_unique_models"] = float(len(model_counts))
    stats["batch_unique_named_models"] = float(len(named_models))
    stats["batch_missing_model_sources"] = float(missing_model_sources)
    stats["batch_model_entropy"] = float(entropy)
    stats["batch_model_perplexity"] = float(math.exp(entropy)) if total_items > 0 else 0.0
    return stats



def _fetch_presliced_training_batch_cpu(
    *,
    dataset_iter: Iterator[Any] | None,
    batch_size: int,
    logger: logging.Logger,
) -> PreparedTrainingBatch:
    if dataset_iter is None:
        raise RuntimeError("presliced dataset iterator is required")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    build_t0 = time.perf_counter()
    records = [_next_valid_presliced_slice(dataset_iter, logger) for _ in range(int(batch_size))]
    first = records[0]
    target_x_shape = tuple(first.x.shape)
    target_W_shape = tuple(first.W.shape)
    for record in records[1:]:
        if tuple(record.x.shape) != target_x_shape or tuple(record.W.shape) != target_W_shape:
            raise ValueError(
                "Presliced records in one batch must have identical fixed shapes, got "
                f"x={tuple(record.x.shape)} W={tuple(record.W.shape)} expected x={target_x_shape} W={target_W_shape}"
            )
    present_logical_indices = [record.logical_index for record in records if record.logical_index is not None]
    if present_logical_indices and len(present_logical_indices) != len(records):
        raise RuntimeError("prepared batch mixes samples with and without logical stream identities")
    logical_indices = tuple(int(value) for value in present_logical_indices)
    if logical_indices and logical_indices != tuple(range(logical_indices[0], logical_indices[0] + len(records))):
        raise RuntimeError(f"prepared batch logical indices are not contiguous: {logical_indices}")

    return PreparedTrainingBatch(
        W=torch.stack([record.W for record in records], dim=0).contiguous(),
        x=torch.stack([record.x for record in records], dim=0).contiguous(),
        x_mask=torch.stack([record.x_mask for record in records], dim=0).contiguous(),
        d_in_mask=torch.stack([record.d_in_mask for record in records], dim=0).contiguous(),
        d_out_mask=torch.stack([record.d_out_mask for record in records], dim=0).contiguous(),
        source_diversity=_compute_presliced_batch_source_diversity_stats(records),
        build_time_s=time.perf_counter() - build_t0,
        logical_indices=logical_indices,
    )


def _prepared_batch_prefetch_blockers(
    *,
    requested: bool,
    prepared_source_enabled: bool,
    fixed_training_batch_enabled: bool,
    synthetic_layer_enabled: bool,
    use_broadcast: bool,
) -> list[str]:
    """Return blockers for the shared presliced/operator-bank prefetch path."""

    reasons: list[str] = []
    if not requested:
        reasons.append("disabled_by_config")
    if not prepared_source_enabled:
        reasons.append("offline_or_operator_bank_disabled")
    if fixed_training_batch_enabled:
        reasons.append("fixed_training_batch_enabled")
    if synthetic_layer_enabled:
        reasons.append("synthetic_layer_enabled")
    if use_broadcast:
        reasons.append("distributed_broadcast_mode")
    return reasons


__all__ = [
    '_prepare_cpu_bool_mask',
    '_next_valid_presliced_slice',
    '_compute_presliced_batch_source_diversity_stats',
    '_fetch_presliced_training_batch_cpu',
    '_prepared_batch_prefetch_blockers',
]
