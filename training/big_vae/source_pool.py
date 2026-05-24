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

from training.big_vae.batch_padding import _slice_sample
from training.big_vae.data_types import SourceSampleRecord, SourceSliceState



def _build_without_replacement_index_groups(
    *,
    num_items: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...] | None:
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if group_size >= num_items:
        return None

    num_groups = num_items // group_size
    order = torch.randperm(num_items, device=device)
    groups: list[torch.Tensor] = []
    for group_idx in range(num_groups):
        group = order[group_idx * group_size : (group_idx + 1) * group_size].sort().values
        groups.append(group)
    return tuple(groups)



def _make_source_slice_state(
    record: SourceSampleRecord,
    *,
    max_T_patches: int,
    max_d_out: int,
    patch_size: int,
) -> SourceSliceState:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    d_in, d_out = record.W.shape
    T_total = d_in // patch_size
    T_use = min(max_T_patches, T_total)
    d_out_use = min(max_d_out, d_out)
    if T_total <= 0 or T_use <= 0 or d_out_use <= 0:
        raise ValueError(
            "source sample cannot produce training slices with current curriculum: "
            f"W_shape={tuple(record.W.shape)} patch_size={patch_size} max_T_patches={max_T_patches} max_d_out={max_d_out}"
        )

    return SourceSliceState(
        source=record,
        patch_size=int(patch_size),
        row_patch_groups=_build_without_replacement_index_groups(
            num_items=int(T_total),
            group_size=int(T_use),
            device=record.W.device,
        ),
        col_groups=_build_without_replacement_index_groups(
            num_items=int(d_out),
            group_size=int(d_out_use),
            device=record.W.device,
        ),
    )



def _remaining_source_state_slices(state: SourceSliceState) -> int:
    capacities: list[int] = []
    if state.row_patch_groups is not None:
        capacities.append(len(state.row_patch_groups) - int(state.row_cursor))
    if state.col_groups is not None:
        capacities.append(len(state.col_groups) - int(state.col_cursor))
    if not capacities:
        return 0 if state.single_unsliced_consumed else 1
    return max(0, min(capacities))



def _source_state_slice_shape(state: SourceSliceState) -> tuple[int, int]:
    if state.row_patch_groups is not None:
        if len(state.row_patch_groups) <= 0:
            raise ValueError("row_patch_groups must be non-empty when present")
        d_in = int(state.row_patch_groups[0].numel()) * int(state.patch_size)
    else:
        d_in = int(state.source.W.shape[0])

    if state.col_groups is not None:
        if len(state.col_groups) <= 0:
            raise ValueError("col_groups must be non-empty when present")
        d_out = int(state.col_groups[0].numel())
    else:
        d_out = int(state.source.W.shape[1])

    return d_in, d_out



def _source_state_shape_capacities(
    source_states: Sequence[SourceSliceState] | None,
) -> dict[tuple[int, int], int]:
    capacities: dict[tuple[int, int], int] = {}
    if not source_states:
        return capacities

    for state in source_states:
        remaining = _remaining_source_state_slices(state)
        if remaining <= 0:
            continue
        shape = _source_state_slice_shape(state)
        capacities[shape] = capacities.get(shape, 0) + int(remaining)
    return capacities



def _max_source_state_shape_capacity(source_states: Sequence[SourceSliceState] | None) -> int:
    capacities = _source_state_shape_capacities(source_states)
    if not capacities:
        return 0
    return max(int(value) for value in capacities.values())



def _dominant_source_state_shape(
    source_states: Sequence[SourceSliceState] | None,
    *,
    start_offset: int = 0,
) -> tuple[int, int] | None:
    capacities = _source_state_shape_capacities(source_states)
    if not capacities:
        return None

    if not source_states:
        return next(iter(capacities.keys()))

    best_shape: tuple[int, int] | None = None
    best_capacity = -1
    normalized_offset = int(start_offset) % len(source_states)
    for step in range(len(source_states)):
        state = source_states[(normalized_offset + step) % len(source_states)]
        if _remaining_source_state_slices(state) <= 0:
            continue
        shape = _source_state_slice_shape(state)
        capacity = int(capacities.get(shape, 0))
        if capacity > best_capacity:
            best_shape = shape
            best_capacity = capacity
    if best_shape is not None:
        return best_shape
    return next(iter(capacities.keys()))



def _prune_source_states_to_shape(
    source_states: Sequence[SourceSliceState] | None,
    *,
    target_shape: tuple[int, int],
) -> list[SourceSliceState]:
    if not source_states:
        return []
    return [
        state
        for state in source_states
        if _remaining_source_state_slices(state) > 0 and _source_state_slice_shape(state) == target_shape
    ]



def _source_states_total_remaining_slices(source_states: Sequence[SourceSliceState] | None) -> int:
    if not source_states:
        return 0
    return sum(_remaining_source_state_slices(state) for state in source_states)



def _prune_exhausted_source_states(source_states: list[SourceSliceState] | None) -> list[SourceSliceState]:
    if not source_states:
        return []
    return [state for state in source_states if _remaining_source_state_slices(state) > 0]



def _prune_exhausted_source_states_with_offset(
    source_states: list[SourceSliceState] | None,
    start_offset: int,
) -> tuple[list[SourceSliceState], int]:
    if not source_states:
        return [], 0

    survivors: list[SourceSliceState] = []
    remapped_offset = 0
    normalized_offset = int(start_offset)
    for idx, state in enumerate(source_states):
        if _remaining_source_state_slices(state) <= 0:
            continue
        if idx < normalized_offset:
            remapped_offset += 1
        survivors.append(state)

    if not survivors:
        return [], 0
    return survivors, remapped_offset % len(survivors)



def _source_states_uniqueness_keys(
    source_states: Sequence[SourceSliceState] | None,
    *,
    uniqueness: str,
) -> set[str]:
    keys: set[str] = set()
    if not source_states:
        return keys
    for state in source_states:
        key = _source_sample_uniqueness_key(state.source, uniqueness)
        if key is not None:
            keys.add(key)
    return keys



def _consume_slice_from_source_state(state: SourceSliceState) -> tuple[torch.Tensor, torch.Tensor] | None:
    remaining = _remaining_source_state_slices(state)
    if remaining <= 0:
        return None

    W_i = state.source.W
    x_i = state.source.x

    if state.row_patch_groups is not None:
        patch_idx = state.row_patch_groups[state.row_cursor]
        state.row_cursor += 1
        offsets = torch.arange(state.patch_size, device=W_i.device)
        row_idx = (patch_idx.unsqueeze(1) * state.patch_size + offsets.unsqueeze(0)).flatten()
        row_idx = row_idx.clamp(max=int(W_i.shape[0]) - 1)
        W_i = W_i[row_idx, :]
        x_i = x_i[:, row_idx]

    if state.col_groups is not None:
        col_idx = state.col_groups[state.col_cursor]
        state.col_cursor += 1
        W_i = W_i[:, col_idx]

    if state.row_patch_groups is None and state.col_groups is None:
        state.single_unsliced_consumed = True

    return W_i, x_i


__all__ = [
    '_build_without_replacement_index_groups',
    '_make_source_slice_state',
    '_remaining_source_state_slices',
    '_source_state_slice_shape',
    '_source_state_shape_capacities',
    '_max_source_state_shape_capacity',
    '_dominant_source_state_shape',
    '_prune_source_states_to_shape',
    '_source_states_total_remaining_slices',
    '_prune_exhausted_source_states',
    '_prune_exhausted_source_states_with_offset',
    '_source_states_uniqueness_keys',
    '_consume_slice_from_source_state',
]
