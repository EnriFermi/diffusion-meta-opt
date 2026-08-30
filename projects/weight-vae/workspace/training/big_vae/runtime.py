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
from training.big_vae.model_config import build_big_vae_model_config



def _promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return

    expected_sections = (
        "data",
        "collector",
        "streaming",
        "train",
        "model",
        "training_artifacts",
        "logging",
        "hf",
        "models",
    )
    with open_dict(cfg):
        for section in expected_sections:
            if section in cfg:
                continue
            if section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]



def _logger(name: str, rank: int) -> logging.Logger:
    return get_rank_logger(name, rank)



def _seed_everything(seed: int, *, active_cuda_device: torch.device | None = None) -> None:
    runtime_seed_everything(seed, active_cuda_device=active_cuda_device)



def _find_free_port() -> int:
    return runtime_find_free_port()



def _resolve_world_size(cfg: DictConfig) -> int:
    return runtime_resolve_world_size(cfg, section="train", error_prefix="train")



def _resolve_backend(cfg: DictConfig, device: torch.device) -> str:
    return runtime_resolve_backend(cfg, device, section="train")



def _resolve_device(cfg: DictConfig, rank: int, world_size: int) -> torch.device:
    return runtime_resolve_device(cfg, rank, world_size, section="train")



def _set_speed_optimizations(cfg: DictConfig, device: torch.device) -> None:
    runtime_set_speed_optimizations(cfg, device, section="train")



def _build_model_cfg(cfg: DictConfig) -> ModelConfig:
    return build_big_vae_model_config(cfg)



def _maybe_compile(model: torch.nn.Module, cfg: DictConfig, logger: logging.Logger) -> torch.nn.Module:
    train_cfg = cfg.train
    compile_dynamic = bool(train_cfg.get("compile_dynamic", True))
    slice_batch_size = max(1, int(train_cfg.get("slice_batch_size", 1)))
    max_safe_slice_batch_size = int(train_cfg.get("compile_dynamic_max_safe_slice_batch_size", 128))
    if compile_dynamic and max_safe_slice_batch_size > 0 and slice_batch_size > max_safe_slice_batch_size:
        logger.warning(
            "Disabling train.compile_dynamic at runtime because train.slice_batch_size=%s exceeds "
            "train.compile_dynamic_max_safe_slice_batch_size=%s; large BigVAE batches can hit "
            "torch.compile backward CantSplit / Inductor failures in dynamic mode",
            slice_batch_size,
            max_safe_slice_batch_size,
        )
        with open_dict(cfg):
            cfg.train.compile_dynamic = False
    return maybe_compile_model(model, cfg, logger, section="train", label="model")



def _build_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    return build_adamw_optimizer(
        model=model,
        cfg=cfg,
        device=device,
        section="train",
        default_lr=3e-4,
        default_weight_decay=0.01,
    )



def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR | None:
    return build_cosine_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        section="train",
        default_max_steps=1000,
        default_warmup_steps=100,
        default_min_lr_ratio=0.1,
    )



def _compute_kl_beta_for_step(
    global_step: int,
    *,
    target_beta: float,
    schedule_enabled: bool,
    start_beta: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    if not schedule_enabled:
        return float(target_beta)
    if global_step <= max(0, int(warmup_steps)):
        return float(start_beta)
    if ramp_steps <= 0:
        return float(target_beta)
    progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(ramp_steps)))
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(start_beta + (target_beta - start_beta) * cosine_progress)



def _compute_latent_sampling_gate_for_step(
    global_step: int,
    *,
    schedule_enabled: bool,
    start_step: int,
    ramp_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    if not schedule_enabled:
        return float(end_value)
    if global_step <= max(0, int(start_step)):
        return float(start_value)
    if ramp_steps <= 0:
        return float(end_value)
    progress = min(1.0, max(0.0, float(global_step - start_step) / float(ramp_steps)))
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    value = float(start_value + (end_value - start_value) * cosine_progress)
    return max(0.0, min(1.0, value))


def _unwrap_model_for_runtime_state(model: torch.nn.Module) -> torch.nn.Module:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if isinstance(compiled_target, torch.nn.Module):
        target = compiled_target
    return target



def _set_model_latent_sampling_gate(model: torch.nn.Module, gate: float) -> None:
    target = _unwrap_model_for_runtime_state(model)
    setter = getattr(target, "set_latent_sampling_gate", None)
    if callable(setter):
        setter(float(gate))



def _compute_model_latent_kl(model: torch.nn.Module, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    target = _unwrap_model_for_runtime_state(model)
    latent_kl = getattr(target, "latent_kl_loss", None)
    if callable(latent_kl):
        return latent_kl(mu, logvar)
    return WeightQuantileVAE.kl_loss(mu, logvar)



def _resolve_amp(cfg: DictConfig, device: torch.device) -> tuple[bool, torch.dtype | None]:
    return runtime_resolve_amp(cfg, device, section="train")



def _autocast_context(enabled: bool, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    return runtime_autocast_context(enabled, dtype)



def _configure_run_artifacts(cfg: DictConfig) -> dict[str, str]:
    return runtime_configure_per_run_artifacts(cfg, run_label="train_big_vae")

__all__ = [
    '_promote_run_profile_to_root',
    '_logger',
    '_seed_everything',
    '_find_free_port',
    '_resolve_world_size',
    '_resolve_backend',
    '_resolve_device',
    '_set_speed_optimizations',
    '_build_model_cfg',
    '_maybe_compile',
    '_build_optimizer',
    '_build_scheduler',
    '_compute_kl_beta_for_step',
    '_compute_latent_sampling_gate_for_step',
    '_unwrap_model_for_runtime_state',
    '_set_model_latent_sampling_gate',
    '_compute_model_latent_kl',
    '_resolve_amp',
    '_autocast_context',
    '_configure_run_artifacts',
]
