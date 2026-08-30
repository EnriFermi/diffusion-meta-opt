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

from training.big_vae.checkpointing import _strip_ddp_prefix



def _layer_key_from_param_name(name: str) -> str:
    parts = name.split(".")
    if not parts:
        return name
    last = parts[-1]
    if last in {"weight", "bias"} or last.endswith("_weight") or last.endswith("_bias"):
        if len(parts) > 1:
            return ".".join(parts[:-1])
    return name



def _grad_stat_group_prefixes() -> dict[str, tuple[str, ...]]:
    return {
        "distribution_encoder": ("distribution_encoder.",),
        "patch_tokenizer": ("patch_tokenizer.", "patch_token_proj.", "cls_token"),
        "encoder": (
            "encoder_layers.",
            "latent_resampler_layers.",
            "enc_dist_inject_projs.",
            "enc_dist_to_latent_heads.",
            "encoder_conditioning_adapters.",
            "latent_base",
            "vamp_prior_base",
            "latent_norm.",
            "to_mu.",
            "to_logvar.",
        ),
        "decoder": (
            "decoder_layers.",
            "mandatory_latent_bridge.",
            "latent_to_decoder.",
            "pos_proj.",
            "query_pos_proj.",
            "query_proj.",
            "direction_head.",
            "z_shortcut_proj.",
            "z_shortcut.",
            "scale_head.",
        ),
        "big_vae_other": tuple(),
    }



def _grad_group_name_for_param(name: str, groups: dict[str, tuple[str, ...]]) -> str:
    for group_name, prefixes in groups.items():
        if group_name == "big_vae_other":
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            return group_name
    return "big_vae_other"



def _parameter_count_summary(model: nn.Module) -> tuple[int, int, int]:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = int(param.numel())
        total_params += count
        if param.requires_grad:
            trainable_params += count
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params



def _collect_params_by_grad_group(model: nn.Module, groups: dict[str, tuple[str, ...]]) -> dict[str, list[nn.Parameter]]:
    out: dict[str, list[nn.Parameter]] = {group_name: [] for group_name in groups}
    for raw_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name = _strip_ddp_prefix(raw_name)
        group_name = _grad_group_name_for_param(name, groups)
        out[group_name].append(param)
    return out



def _grad_l2_norm_for_params(params: list[nn.Parameter]) -> float:
    sum_sq = 0.0
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        g = grad.detach()
        sum_sq += float((g * g).sum().item())
    return math.sqrt(sum_sq)



def _clip_grad_norm_with_optional_foreach(
    get_params: Callable[[], Any],
    max_norm: float,
    *,
    use_foreach: bool,
) -> Any:
    """Use foreach clipping when available; gracefully fallback on older torch."""
    try:
        return torch.nn.utils.clip_grad_norm_(get_params(), max_norm, foreach=use_foreach)
    except TypeError:
        return torch.nn.utils.clip_grad_norm_(get_params(), max_norm)



def _clip_return_to_float(clip_return: Any) -> float:
    if torch.is_tensor(clip_return):
        return float(clip_return.detach().item())
    return float(clip_return)



def compute_grad_stats(model: nn.Module) -> dict[str, float]:
    groups = _grad_stat_group_prefixes()
    group_sums = {name: 0.0 for name in groups}
    group_numel = {name: 0 for name in groups}
    group_param_count = {name: 0 for name in groups}

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    grad_numel = 0
    param_sum_sq = 0.0
    param_numel = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        if not param.requires_grad:
            continue

        p = param.detach()
        param_sum_sq += float(p.pow(2).sum().item())
        param_numel += int(p.numel())

        grad = param.grad
        if grad is None:
            continue

        g = grad.detach()
        abs_g = g.abs()
        sum_sq += float((g * g).sum().item())
        sum_abs += float(abs_g.sum().item())
        max_abs = max(max_abs, float(abs_g.max().item()))
        numel = int(g.numel())
        grad_numel += numel

        group_name = _grad_group_name_for_param(name, groups)
        group_param_count[group_name] += 1
        group_sums[group_name] += float((g * g).sum().item())
        group_numel[group_name] += numel

    grad_rms = math.sqrt(sum_sq / max(1, grad_numel))
    param_rms = math.sqrt(param_sum_sq / max(1, param_numel))
    payload: dict[str, float] = {
        "grad/global_norm": math.sqrt(sum_sq),
        "grad/rms": grad_rms,
        "grad/abs_mean": (sum_abs / max(1, grad_numel)),
        "grad/max_abs": max_abs,
        "grad/numel": float(grad_numel),
        "param/rms": param_rms,
        "grad_to_param_rms_ratio": grad_rms / max(1e-12, param_rms),
    }

    for group_name in groups:
        payload[f"grad/{group_name}_rms"] = (
            math.sqrt(group_sums[group_name] / max(1, group_numel[group_name])) if group_numel[group_name] > 0 else 0.0
        )
        payload[f"grad/{group_name}_numel"] = float(group_numel[group_name])
        payload[f"grad/{group_name}_params_with_grad"] = float(group_param_count[group_name])

    return payload



def _collect_nonfinite_grad_report(
    model: nn.Module,
    *,
    max_items: int = 12,
) -> dict[str, Any] | None:
    groups = _grad_stat_group_prefixes()
    total_bad_tensors = 0
    total_nan = 0
    total_inf = 0
    items: list[dict[str, Any]] = []

    for raw_name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        grad = param.grad.detach()
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all().item()):
            continue

        total_bad_tensors += 1
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        total_nan += nan_count
        total_inf += inf_count

        if len(items) >= int(max_items):
            continue

        name = _strip_ddp_prefix(raw_name)
        item: dict[str, Any] = {
            "name": name,
            "layer": _layer_key_from_param_name(name),
            "group": _grad_group_name_for_param(name, groups),
            "shape": list(grad.shape),
            "dtype": str(grad.dtype),
            "device": str(grad.device),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_count": int(finite_mask.sum().item()),
        }
        if item["finite_count"] > 0:
            finite_vals = grad[finite_mask]
            item["finite_abs_max"] = float(finite_vals.abs().max().item())
            item["finite_mean"] = float(finite_vals.mean().item())
            item["finite_std"] = float(finite_vals.std(unbiased=False).item())
        items.append(item)

    if total_bad_tensors == 0:
        return None

    return {
        "bad_tensors": int(total_bad_tensors),
        "total_nan": int(total_nan),
        "total_inf": int(total_inf),
        "items": items,
    }



def collect_grad_rms_per_layer(
    model: nn.Module,
    *,
    include_prefixes: tuple[str, ...],
    weights_only: bool,
) -> dict[str, float]:
    sumsq_by_layer: dict[str, float] = {}
    count_by_layer: dict[str, int] = {}
    global_sumsq = 0.0
    global_count = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue
        if weights_only and param.ndim < 2:
            continue

        grad = param.grad
        if grad is None:
            continue
        g = grad.detach()
        if g.numel() == 0:
            continue

        sumsq = float((g * g).sum().item())
        count = int(g.numel())
        layer = _layer_key_from_param_name(name)
        sumsq_by_layer[layer] = sumsq_by_layer.get(layer, 0.0) + sumsq
        count_by_layer[layer] = count_by_layer.get(layer, 0) + count

        global_sumsq += sumsq
        global_count += count

    out: dict[str, float] = {}
    for layer, sumsq in sumsq_by_layer.items():
        out[layer] = math.sqrt(sumsq / float(max(1, count_by_layer[layer])))
    if global_count > 0:
        out["__global__"] = math.sqrt(global_sumsq / float(global_count))
    return out



def collect_param_rms_per_layer(
    model: nn.Module,
    *,
    include_prefixes: tuple[str, ...],
    weights_only: bool,
) -> dict[str, float]:
    sumsq_by_layer: dict[str, float] = {}
    count_by_layer: dict[str, int] = {}
    global_sumsq = 0.0
    global_count = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue
        if weights_only and param.ndim < 2:
            continue

        p = param.detach()
        if p.numel() == 0:
            continue

        sumsq = float((p * p).sum().item())
        count = int(p.numel())
        layer = _layer_key_from_param_name(name)
        sumsq_by_layer[layer] = sumsq_by_layer.get(layer, 0.0) + sumsq
        count_by_layer[layer] = count_by_layer.get(layer, 0) + count
        global_sumsq += sumsq
        global_count += count

    out: dict[str, float] = {}
    for layer, sumsq in sumsq_by_layer.items():
        out[layer] = math.sqrt(sumsq / float(max(1, count_by_layer[layer])))
    if global_count > 0:
        out["__global__"] = math.sqrt(global_sumsq / float(global_count))
    return out


__all__ = [
    '_layer_key_from_param_name',
    '_grad_stat_group_prefixes',
    '_grad_group_name_for_param',
    '_parameter_count_summary',
    '_collect_params_by_grad_group',
    '_grad_l2_norm_for_params',
    '_clip_grad_norm_with_optional_foreach',
    '_clip_return_to_float',
    'compute_grad_stats',
    '_collect_nonfinite_grad_report',
    'collect_grad_rms_per_layer',
    'collect_param_rms_per_layer',
]
