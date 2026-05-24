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



def _seed_everything(seed: int) -> None:
    runtime_seed_everything(seed)



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
    model_cfg = cfg.get("model", {})

    dist_cfg = model_cfg.get("distribution", {})
    mini_cfg = model_cfg.get("mini_vae", {})
    big_cfg = model_cfg.get("big_vae", {})
    enc_cfg = big_cfg.get("encoder", {})

    return ModelConfig(
        patch_size=int(model_cfg.get("patch_size", 16)),
        beta=float(model_cfg.get("beta", 1e-3)),
        variant=str(model_cfg.get("variant", "full")),
        distribution=DistributionConfig(
            k_s=int(dist_cfg.get("k_s", 16)),
            Kq=int(dist_cfg.get("Kq", 32)),
            d_var=int(dist_cfg.get("d_var", 128)),
            d_dist=int(dist_cfg.get("d_dist", 128)),
            num_var_attn_layers=int(dist_cfg.get("num_var_attn_layers", 2)),
            var_attn_heads=int(dist_cfg.get("var_attn_heads", 4)),
            dcn_num_cross_layers=int(dist_cfg.get("dcn_num_cross_layers", 3)),
            dcn_deep_hidden=int(dist_cfg.get("dcn_deep_hidden", 0)),
            dcn_deep_layers=int(dist_cfg.get("dcn_deep_layers", 0)),
            dropout=float(dist_cfg.get("dropout", 0.0)),
            use_covariance=bool(dist_cfg.get("use_covariance", True)),
            patch_size_for_cov=int(dist_cfg.get("patch_size_for_cov", int(model_cfg.get("patch_size", 16)))),
        ),
        mini_vae=MiniVAEConfig(
            z_dim=int(mini_cfg.get("z_dim", 64)),
            d_e=int(mini_cfg.get("d_e", 128)),
            num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
            num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
            decoder_bilinear_rank=int(mini_cfg.get("decoder_bilinear_rank", 0)),
            stub_resampler_d_model=int(mini_cfg.get("stub_resampler_d_model", 0)),
            init_style=str(mini_cfg.get("init_style", "llm")),
            n_heads=int(mini_cfg.get("n_heads", 4)),
            d_patch=int(mini_cfg.get("d_patch", 64)),
            dropout=float(mini_cfg.get("dropout", 0.0)),
        ),
        big_vae=BigVAEConfig(
            d_model=int(big_cfg.get("d_model", 256)),
            d_lat=int(big_cfg.get("d_lat", 256)),
            num_latents=int(big_cfg.get("num_latents", 32)),
            num_encoder_layers=int(big_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(big_cfg.get("num_decoder_layers", 4)),
            n_heads=int(big_cfg.get("n_heads", 8)),
            ffn_mult=float(big_cfg.get("ffn_mult", 4.0)),
            dropout=float(big_cfg.get("dropout", 0.0)),
            pos_fourier_dim=int(big_cfg.get("pos_fourier_dim", 64)),
            use_latent_sampling=bool(big_cfg.get("use_latent_sampling", True)),
            use_encoder_mu_head=bool(big_cfg.get("use_encoder_mu_head", False)),
            normalize_latent_slots_before_mu=bool(big_cfg.get("normalize_latent_slots_before_mu", True)),
            latent_prior_kind=str(big_cfg.get("latent_prior_kind", "gaussian")),
            vamp_prior_K=int(big_cfg.get("vamp_prior_K", 64)),
            decoder_query_conditioning_kind=str(big_cfg.get("decoder_query_conditioning_kind", "linear")),
            decoder_query_conditioning_hidden_mult=float(big_cfg.get("decoder_query_conditioning_hidden_mult", 2.0)),
            rope_2d_coord_kind=str(big_cfg.get("rope_2d_coord_kind", "normalized_center")),
            latent_sampling_min_std=float(big_cfg.get("latent_sampling_min_std", 1e-4)),
            latent_sampling_logvar_min=float(big_cfg.get("latent_sampling_logvar_min", -20.0)),
            latent_sampling_logvar_max=float(big_cfg.get("latent_sampling_logvar_max", 10.0)),
            disable_z_shortcut=bool(big_cfg.get("disable_z_shortcut", False)),
            disable_distribution_encoder=bool(big_cfg.get("disable_distribution_encoder", False)),
            patch_tokenizer_kind=str(big_cfg.get("patch_tokenizer_kind", "residual")),
            distribution_encoder_conditioning_kind=str(big_cfg.get("distribution_encoder_conditioning_kind", "legacy")),
            encoder=EncoderConfig(
                self_attn_mode=str(enc_cfg.get("self_attn_mode", "full")),
                cross_attend_only_cls=bool(enc_cfg.get("cross_attend_only_cls", True)),
            ),
        ),
    )



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



def _set_model_latent_sampling_gate(model: torch.nn.Module, gate: float) -> None:
    target = _unwrap_model_for_state_io(model)
    setter = getattr(target, "set_latent_sampling_gate", None)
    if callable(setter):
        setter(float(gate))



def _compute_model_latent_kl(model: torch.nn.Module, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    target = _unwrap_model_for_state_io(model)
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
    '_set_model_latent_sampling_gate',
    '_compute_model_latent_kl',
    '_resolve_amp',
    '_autocast_context',
    '_configure_run_artifacts',
]
