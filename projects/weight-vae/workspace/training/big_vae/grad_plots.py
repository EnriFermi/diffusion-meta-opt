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



def _append_grad_layer_rms_csv(
    save_path: Path,
    *,
    step: int,
    layer_rms_pre_clip: dict[str, float],
    layer_rms_post_clip: dict[str, float],
    layer_param_rms: dict[str, float],
    layer_grad_to_param_ratio_pre_clip: dict[str, float],
    layer_grad_to_param_ratio_post_clip: dict[str, float],
    clip_coef: float,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        save_path.write_text(
            "step,layer,grad_rms_pre_clip,grad_rms_post_clip,param_rms,grad_to_param_ratio_pre_clip,"
            "grad_to_param_ratio_post_clip,clip_coef\n",
            encoding="utf-8",
        )

    layers = sorted(
        set(layer_rms_pre_clip.keys())
        | set(layer_rms_post_clip.keys())
        | set(layer_param_rms.keys())
        | set(layer_grad_to_param_ratio_pre_clip.keys())
        | set(layer_grad_to_param_ratio_post_clip.keys())
    )
    with save_path.open("a", encoding="utf-8") as handle:
        for layer in layers:
            pre_val = float(layer_rms_pre_clip.get(layer, 0.0))
            post_val = float(layer_rms_post_clip.get(layer, 0.0))
            param_val = float(layer_param_rms.get(layer, 0.0))
            ratio_pre_val = float(layer_grad_to_param_ratio_pre_clip.get(layer, 0.0))
            ratio_post_val = float(layer_grad_to_param_ratio_post_clip.get(layer, 0.0))
            handle.write(
                f"{int(step)},{layer},{pre_val:.12e},{post_val:.12e},{param_val:.12e},"
                f"{ratio_pre_val:.12e},{ratio_post_val:.12e},{float(clip_coef):.12e}\n"
            )



def _save_grad_rms_layer_plot(
    history: dict[str, list[tuple[int, float]]],
    save_path: Path,
    *,
    topk_layers: int,
    log_scale: bool,
) -> bool:
    if not history:
        return False
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return False

    layers = [layer for layer in history.keys() if layer != "__global__"]
    layers.sort(
        key=lambda key: sum(val for _, val in history[key]) / max(1, len(history[key])),
        reverse=True,
    )
    layers = layers[: max(1, int(topk_layers))]

    plt.figure(figsize=(13, 6))
    if "__global__" in history:
        steps = [step for step, _ in history["__global__"]]
        values = [value for _, value in history["__global__"]]
        plt.plot(steps, values, label="__global__", linewidth=2.5, color="black")
    for layer in layers:
        steps = [step for step, _ in history[layer]]
        values = [value for _, value in history[layer]]
        plt.plot(steps, values, label=layer, linewidth=1.1, alpha=0.9)

    plt.title("BigVAE Gradient RMS Per Layer (pre-clip)")
    plt.xlabel("Step")
    plt.ylabel("Gradient RMS")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.2)
    if layers or "__global__" in history:
        plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()
    return True



def _save_grad_rms_layer_heatmap(
    history: dict[str, list[tuple[int, float]]],
    save_path: Path,
    *,
    max_layers: int,
    log_scale: bool,
    min_value: float = 1e-20,
) -> bool:
    if not history:
        return False
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return False

    layers = [layer for layer in history.keys() if layer != "__global__"]
    if not layers:
        return False
    layers.sort(key=lambda key: sum(val for _, val in history[key]) / max(1, len(history[key])))
    layers = layers[: max(1, int(max_layers))]

    step_values = sorted({int(step) for points in history.values() for step, _ in points})
    if not step_values:
        return False
    step_to_idx = {step: idx for idx, step in enumerate(step_values)}

    matrix: list[list[float]] = []
    for layer in layers:
        row = [float("nan")] * len(step_values)
        for step, value in history[layer]:
            idx = step_to_idx.get(int(step))
            if idx is None:
                continue
            safe_value = max(float(min_value), float(value))
            row[idx] = math.log10(safe_value) if log_scale else safe_value
        matrix.append(row)

    fig_h = max(4.0, 0.24 * len(layers) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("BigVAE Gradient RMS Heatmap (pre-clip)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Step")
    ax.set_yticks(list(range(len(layers))))
    ax.set_yticklabels(layers, fontsize=7)

    xtick_count = min(12, len(step_values))
    if xtick_count >= 2:
        xtick_idx = sorted(
            set(int(round(i * (len(step_values) - 1) / float(xtick_count - 1))) for i in range(xtick_count))
        )
    else:
        xtick_idx = [0]
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([str(step_values[idx]) for idx in xtick_idx], rotation=45, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(grad RMS)" if log_scale else "grad RMS")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    return True


__all__ = [
    '_append_grad_layer_rms_csv',
    '_save_grad_rms_layer_plot',
    '_save_grad_rms_layer_heatmap',
]
