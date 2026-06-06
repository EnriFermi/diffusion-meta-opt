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

def _strip_ddp_prefix(name: str) -> str:
    out = str(name)
    # DDP + torch.compile wrappers can both prepend module qualifiers.
    changed = True
    while changed:
        changed = False
        if out.startswith("module."):
            out = out[len("module.") :]
            changed = True
        if out.startswith("_orig_mod."):
            out = out[len("_orig_mod.") :]
            changed = True
    return out



def _unwrap_model_for_state_io(model: torch.nn.Module) -> torch.nn.Module:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if isinstance(compiled_target, torch.nn.Module):
        target = compiled_target
    return target



def _normalize_model_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = _strip_ddp_prefix(str(raw_key))
        existing = normalized.get(key)
        if existing is not None and existing is not value:
            raise ValueError(
                f"State dict key collision after normalization: raw_key={raw_key!r} normalized_key={key!r}"
            )
        normalized[key] = value
    return normalized



def _save_checkpoint_payload(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass



def _save_checkpoint(
    model: torch.nn.Module,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int = 1,
) -> None:
    base_dir = Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/weight_quantile_vae")))
    checkpoint_dir = base_dir / f"stage_{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = _unwrap_model_for_state_io(model)
    payload = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    _save_checkpoint_payload(payload, step_path)
    _save_checkpoint_payload(payload, latest_path)
    logger.info("Model checkpoint saved: %s", step_path)



def _save_resume_state_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
    stage: int,
    state_dir: Path,
) -> None:
    model_to_save = _unwrap_model_for_state_io(model)
    payload = {
        "step": step_idx,
        "stage": stage,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    state_path = state_dir / f"step_{step_idx:07d}.pt"
    _save_checkpoint_payload(payload, state_path)

    for stale_path in sorted(state_dir.glob("step_*.pt")):
        if stale_path == state_path:
            continue
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not delete stale resume-state checkpoint %s: %s", stale_path, exc)
    logger.info("Resume-state checkpoint saved: %s", state_path)



def _maybe_dump_fixed_training_batch(
    *,
    W_s: torch.Tensor,
    x_s: torch.Tensor,
    x_mask_s: torch.Tensor | None,
    d_in_mask_s: torch.Tensor | None,
    d_out_mask_s: torch.Tensor | None,
    cfg: DictConfig,
    logger: logging.Logger,
    global_step: int,
    stage: int,
    patch_size: int,
    max_T_patches: int,
    max_d_out: int,
    slice_batch_size: int,
) -> Path | None:
    fixed_batch_cfg = cfg.train.get("fixed_training_batch", {})
    if fixed_batch_cfg is None:
        return None
    if not isinstance(fixed_batch_cfg, (dict, DictConfig)):
        raise TypeError("train.fixed_training_batch must be a mapping")

    dump_path_text = str(fixed_batch_cfg.get("dump_path", "")).strip()
    if not dump_path_text:
        return None

    dump_path = Path(dump_path_text)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fixed_batch_W": W_s.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
        "fixed_batch_x": x_s.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
        "fixed_batch_x_mask": (
            x_mask_s.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
            if x_mask_s is not None
            else None
        ),
        "fixed_batch_d_in_mask": (
            d_in_mask_s.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
            if d_in_mask_s is not None
            else None
        ),
        "fixed_batch_d_out_mask": (
            d_out_mask_s.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous()
            if d_out_mask_s is not None
            else None
        ),
        "meta": {
            "capture_step": int(global_step),
            "stage": int(stage),
            "patch_size": int(patch_size),
            "max_T_patches": int(max_T_patches),
            "max_d_out": int(max_d_out),
            "slice_batch_size": int(slice_batch_size),
            "W_shape": list(W_s.shape),
            "x_shape": list(x_s.shape),
            "x_mask_shape": list(x_mask_s.shape) if x_mask_s is not None else None,
            "d_in_mask_shape": list(d_in_mask_s.shape) if d_in_mask_s is not None else None,
            "d_out_mask_shape": list(d_out_mask_s.shape) if d_out_mask_s is not None else None,
            "data_seed": int(cfg.data.get("seed", 42)),
        },
    }
    torch.save(payload, dump_path)
    logger.info("Fixed training batch dump saved: %s", dump_path)
    return dump_path



def _load_frozen_training_batch_dump(
    dump_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(dump_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen training batch dump not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Frozen training batch dump must contain a dict payload, got {type(payload)!r}: {path}")

    W_s = payload.get("fixed_batch_W", payload.get("W_s"))
    x_s = payload.get("fixed_batch_x", payload.get("x_s"))
    if not torch.is_tensor(W_s) or not torch.is_tensor(x_s):
        raise KeyError(
            f"Frozen training batch dump must contain tensor keys 'fixed_batch_W'/'fixed_batch_x' or 'W_s'/'x_s': {path}"
        )

    W_s = W_s.detach().to(device=device, dtype=torch.float32, copy=True).contiguous()
    x_s = x_s.detach().to(device=device, dtype=torch.float32, copy=True).contiguous()
    if W_s.ndim != 3 or x_s.ndim != 3:
        raise ValueError(
            "Frozen training batch tensors must be rank-3 batched tensors, got "
            f"W={tuple(W_s.shape)} x={tuple(x_s.shape)} from {path}"
        )
    if int(W_s.shape[0]) != int(x_s.shape[0]) or int(W_s.shape[1]) != int(x_s.shape[2]):
        raise ValueError(
            "Frozen training batch tensor shapes are inconsistent, got "
            f"W={tuple(W_s.shape)} x={tuple(x_s.shape)} from {path}"
        )

    raw_meta = payload.get("meta", {})
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {"raw_meta": raw_meta}
    meta.setdefault("loaded_W_shape", list(W_s.shape))
    meta.setdefault("loaded_x_shape", list(x_s.shape))
    meta.setdefault("dump_path", str(path))
    return W_s, x_s, meta



def _model_uses_latent_sampling(model: torch.nn.Module) -> bool:
    cfg = getattr(model, "cfg", None)
    big_vae_cfg = getattr(cfg, "big_vae", None)
    return bool(getattr(big_vae_cfg, "use_latent_sampling", False))



def _model_uses_encoder_mu_head(model: torch.nn.Module) -> bool:
    cfg = getattr(model, "cfg", None)
    big_vae_cfg = getattr(cfg, "big_vae", None)
    return bool(getattr(big_vae_cfg, "use_encoder_mu_head", False))



def _is_vae_posterior_head_key(key: str) -> bool:
    return any(str(key).startswith(prefix) for prefix in _VAE_POSTERIOR_HEAD_PREFIXES)



def _is_allowed_optional_latent_head_key(*, model: torch.nn.Module, key: str) -> bool:
    key_str = str(key)
    if key_str.startswith("to_mu."):
        return _model_uses_latent_sampling(model) or _model_uses_encoder_mu_head(model)
    if key_str.startswith("to_logvar."):
        return _model_uses_latent_sampling(model)
    if key_str.startswith("vamp_prior_base"):
        return True
    return False



def _load_model_state_allowing_vae_head_migration(
    *,
    target: torch.nn.Module,
    state_dict: dict[str, Any],
    logger: logging.Logger,
    source: str | Path,
) -> bool:
    target_state = target.state_dict()
    missing_keys = sorted(key for key in target_state.keys() if key not in state_dict)
    unexpected_keys = sorted(key for key in state_dict.keys() if key not in target_state)

    if not missing_keys and not unexpected_keys:
        target.load_state_dict(state_dict, strict=True)
        return False

    allowed_missing = sorted(key for key in missing_keys if _is_allowed_optional_latent_head_key(model=target, key=key))
    allowed_unexpected = sorted(
        key for key in unexpected_keys if _is_allowed_optional_latent_head_key(model=target, key=key)
    )
    should_allow_latent_head_migration = (
        bool(allowed_missing or allowed_unexpected)
        and allowed_missing == missing_keys
        and allowed_unexpected == unexpected_keys
    )
    if not should_allow_latent_head_migration:
        target.load_state_dict(state_dict, strict=True)
        return False

    incompatible = target.load_state_dict(state_dict, strict=False)
    unexpected_after_load = list(getattr(incompatible, "unexpected_keys", []))
    missing_after_load = sorted(getattr(incompatible, "missing_keys", []))
    disallowed_missing = [key for key in missing_after_load if not _is_allowed_optional_latent_head_key(model=target, key=key)]
    disallowed_unexpected = [
        key for key in unexpected_after_load if not _is_allowed_optional_latent_head_key(model=target, key=key)
    ]
    if disallowed_unexpected or disallowed_missing:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility while migrating optional latent heads: "
            f"source={source} missing={missing_after_load} unexpected={unexpected_after_load}"
        )

    logger.warning(
        "Loaded BigVAE checkpoint with optional latent-head migration. "
        "source=%s missing_head_keys=%s dropped_head_keys=%s",
        source,
        missing_after_load,
        unexpected_after_load,
    )
    return True



def _load_model_weights_from_checkpoint(
    model: torch.nn.Module,
    path: str,
    logger: logging.Logger,
) -> bool:
    logger.info("Loading model weights from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _normalize_model_state_dict_keys(ckpt["model_state"])
    target = _unwrap_model_for_state_io(model)
    migrated_ae_to_vae = _load_model_state_allowing_vae_head_migration(
        target=target,
        state_dict=state_dict,
        logger=logger,
        source=path,
    )
    logger.info("Model weights loaded successfully (step=%s)", ckpt.get("step", "?"))
    return migrated_ae_to_vae



def _find_latest_resume_state_checkpoint(state_dir: Path) -> Path | None:
    if not state_dir.exists() or not state_dir.is_dir():
        return None
    step_paths = sorted(path for path in state_dir.glob("step_*.pt") if path.is_file())
    if step_paths:
        return step_paths[-1]
    latest_path = state_dir / "latest.pt"
    if latest_path.is_file():
        return latest_path
    return None



def _resume_state_load_policy(resume_state_cfg: dict[str, Any] | DictConfig) -> dict[str, bool]:
    return {
        "load_model_state": bool(resume_state_cfg.get("load_model_state", True)),
        "load_optimizer_state": bool(resume_state_cfg.get("load_optimizer_state", True)),
        "load_scheduler_state": bool(resume_state_cfg.get("load_scheduler_state", True)),
        "load_scaler_state": bool(resume_state_cfg.get("load_scaler_state", True)),
        "load_step": bool(resume_state_cfg.get("load_step", True)),
    }



def _load_training_state_from_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    path: Path,
    logger: logging.Logger,
    load_model_state: bool = True,
    load_optimizer_state: bool = True,
    load_scheduler_state: bool = True,
    load_scaler_state: bool = True,
    load_step: bool = True,
) -> int:
    logger.info("Loading training state from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    migrated_ae_to_vae = False
    if load_model_state:
        target = _unwrap_model_for_state_io(model)
        state_dict = _normalize_model_state_dict_keys(ckpt["model_state"])
        migrated_ae_to_vae = _load_model_state_allowing_vae_head_migration(
            target=target,
            state_dict=state_dict,
            logger=logger,
            source=path,
        )

    if migrated_ae_to_vae and load_optimizer_state:
        raise RuntimeError(
            "Cannot load optimizer_state while migrating an AE checkpoint without posterior heads "
            "into use_latent_sampling=true BigVAE. Set train.resume_state.load_optimizer_state=false "
            "and resume model-only for AE -> VAE fine-tuning."
        )
    optimizer_state = ckpt.get("optimizer_state")
    if load_optimizer_state and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = ckpt.get("scheduler_state")
    if load_scheduler_state and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = ckpt.get("scaler_state")
    if load_scaler_state and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    step = int(ckpt.get("step", 0) or 0) if load_step else 0
    stage = ckpt.get("stage", "?")
    logger.info(
        "Training state loaded successfully (path=%s step=%s stage=%s load_model_state=%s "
        "load_optimizer_state=%s load_scheduler_state=%s load_scaler_state=%s load_step=%s)",
        path,
        step,
        stage,
        load_model_state,
        load_optimizer_state,
        load_scheduler_state,
        load_scaler_state,
        load_step,
    )
    return step

__all__ = [
    '_strip_ddp_prefix',
    '_unwrap_model_for_state_io',
    '_normalize_model_state_dict_keys',
    '_save_checkpoint_payload',
    '_save_checkpoint',
    '_save_resume_state_checkpoint',
    '_maybe_dump_fixed_training_batch',
    '_load_frozen_training_batch_dump',
    '_model_uses_latent_sampling',
    '_model_uses_encoder_mu_head',
    '_is_vae_posterior_head_key',
    '_is_allowed_optional_latent_head_key',
    '_load_model_state_allowing_vae_head_migration',
    '_load_model_weights_from_checkpoint',
    '_find_latest_resume_state_checkpoint',
    '_resume_state_load_policy',
    '_load_training_state_from_checkpoint',
]
