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



def _build_collector_status_snapshot(collector: Any) -> dict[str, Any]:
    stats = collector.stats()
    payload: dict[str, Any] = {
        "mode": stats.get("mode"),
        "streaming_mode": stats.get("streaming_mode"),
        "parallel_inference_workers": int(stats.get("parallel_inference_workers", 1) or 1),
        "cache_size": int(stats.get("cache_size", 0)),
        "jobs_total": int(stats.get("jobs_total", 0)),
        "items_emitted": int(stats.get("items_emitted", 0)),
        "async_process_alive": stats.get("async_process_alive"),
        "async_events_dropped": stats.get("async_events_dropped"),
        "worker_status_dir": stats.get("worker_status_dir"),
    }

    async_last_status = stats.get("async_last_status")
    if not isinstance(async_last_status, dict):
        return payload

    worker_total = 0
    worker_alive = 0
    worker_disabled = 0
    degraded_workers: list[dict[str, Any]] = []
    raw_workers = async_last_status.get("raw_pool_workers")
    if isinstance(raw_workers, dict):
        for dataset_name, info in raw_workers.items():
            if not isinstance(info, dict):
                continue
            worker_total += 1
            is_alive = bool(info.get("worker_alive", False))
            is_disabled = bool(info.get("disabled", False) or info.get("worker_permanently_stopped", False))
            if is_alive:
                worker_alive += 1
            if is_disabled:
                worker_disabled += 1
            if (not is_alive) or is_disabled or bool(info.get("worker_last_error")):
                degraded_workers.append(
                    {
                        "dataset": str(dataset_name),
                        "worker_pid": info.get("worker_pid"),
                        "worker_alive": is_alive,
                        "disabled": is_disabled,
                        "worker_restarts": int(info.get("worker_restarts", 0) or 0),
                        "worker_last_error": info.get("worker_last_error"),
                    }
                )

    if worker_total > 0:
        payload["dataset_workers_total"] = int(worker_total)
        payload["dataset_workers_alive"] = int(worker_alive)
        payload["dataset_workers_disabled_or_stopped"] = int(worker_disabled)
    if degraded_workers:
        payload["dataset_workers_degraded"] = degraded_workers[:8]

    job_stats = async_last_status.get("job_stats")
    if isinstance(job_stats, dict):
        payload["last_job"] = {
            "model_name": str(job_stats.get("model_name", "")),
            "num_images": int(job_stats.get("num_images", 0)),
            "num_layers": int(job_stats.get("num_layers", 0)),
            "num_samples_emitted": int(job_stats.get("num_samples_emitted", 0)),
            "duration_s": float(job_stats.get("duration_s", 0.0)),
            "raw_batch_fetch_s": float(job_stats.get("raw_batch_fetch_s", 0.0)),
            "model_infer_s": float(job_stats.get("model_infer_s", 0.0)),
            "atomize_emit_s": float(job_stats.get("atomize_emit_s", 0.0)),
        }

    return payload



def _collector_tracked_children(collector: Any) -> list[dict[str, Any]]:
    tracked: list[dict[str, Any]] = []
    try:
        stats = collector.stats()
    except Exception:
        return tracked

    async_last_status = stats.get("async_last_status")
    if not isinstance(async_last_status, dict):
        return tracked

    collector_pid = async_last_status.get("collector_pid")
    if isinstance(collector_pid, int) and collector_pid > 0:
        tracked.append(
            {
                "pid": int(collector_pid),
                "role": "collector_process",
                "metadata": {
                    "mode": stats.get("mode"),
                    "streaming_mode": stats.get("streaming_mode"),
                },
            }
        )

    raw_workers = async_last_status.get("raw_pool_workers")
    if isinstance(raw_workers, dict):
        for dataset_name, info in raw_workers.items():
            if not isinstance(info, dict):
                continue
            worker_pid = info.get("worker_pid")
            if not isinstance(worker_pid, int) or worker_pid <= 0:
                continue
            tracked.append(
                {
                    "pid": int(worker_pid),
                    "role": "dataset_worker",
                    "metadata": {
                        "dataset": str(dataset_name),
                        "worker_alive": bool(info.get("worker_alive", False)),
                        "worker_restarts": int(info.get("worker_restarts", 0) or 0),
                        "worker_last_error": info.get("worker_last_error"),
                    },
                }
            )
    return tracked



def _get_encoder_conditioning_alpha_values(model: nn.Module) -> list[float]:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    adapters = getattr(target, "encoder_conditioning_adapters", None)
    if adapters is None:
        return []

    alpha_values: list[float] = []
    for adapter in adapters:
        alpha = getattr(adapter, "alpha", None)
        if alpha is None or not torch.is_tensor(alpha) or alpha.numel() == 0:
            continue
        alpha_values.append(float(alpha.detach().reshape(-1)[0].item()))
    return alpha_values



def _get_patch_tokenizer_block_alpha_stats(model: nn.Module) -> list[dict[str, float]]:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    patch_tokenizer = getattr(target, "patch_tokenizer", None)
    blocks = getattr(patch_tokenizer, "blocks", None)
    if blocks is None:
        return []

    out: list[dict[str, float]] = []
    for block in blocks:
        alpha = getattr(block, "alpha", None)
        if alpha is None or not torch.is_tensor(alpha):
            continue
        alpha_flat = alpha.detach().reshape(-1).to(dtype=torch.float32)
        if alpha_flat.numel() == 0:
            continue
        out.append(
            {
                "mean": float(alpha_flat.mean().item()),
                "abs_mean": float(alpha_flat.abs().mean().item()),
                "max_abs": float(alpha_flat.abs().max().item()),
            }
        )
    return out



def _get_patch_latent_variance_stats(
    model: nn.Module,
    W: torch.Tensor,
    X: torch.Tensor,
    x_mask: torch.Tensor | None = None,
    d_in_mask: torch.Tensor | None = None,
    d_out_mask: torch.Tensor | None = None,
) -> dict[str, float] | None:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    forward_debug = getattr(target, "forward_debug", None)
    if not callable(forward_debug):
        return None

    was_training = bool(target.training)
    try:
        target.eval()
        with torch.no_grad():
            outputs = forward_debug(W, X, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask)
        if not outputs or not isinstance(outputs[-1], dict):
            return None
        debug_info = outputs[-1]
        dist_var_by_patch = debug_info.get("dist_var_by_patch")
        if not torch.is_tensor(dist_var_by_patch):
            return None
        dist_var_flat = dist_var_by_patch.detach().to(dtype=torch.float32).flatten(start_dim=2)
        if dist_var_flat.numel() == 0:
            return None
        patch_var = dist_var_flat.var(dim=-1, unbiased=False)
        return {
            "mean": float(patch_var.mean().item()),
            "std": float(patch_var.std(unbiased=False).item()),
            "min": float(patch_var.min().item()),
            "max": float(patch_var.max().item()),
        }
    finally:
        target.train(was_training)


__all__ = [
    '_build_collector_status_snapshot',
    '_collector_tracked_children',
    '_get_encoder_conditioning_alpha_values',
    '_get_patch_tokenizer_block_alpha_stats',
    '_get_patch_latent_variance_stats',
]
