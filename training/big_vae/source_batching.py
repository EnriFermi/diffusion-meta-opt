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

from training.big_vae.batch_padding import _materialize_padded_slice_batch, _pad_x_rows_with_mask, _round_robin_source_indices, _sample_synthetic_layer, _slice_sample
from training.big_vae.data_types import ConsumedSourceBatch, SourceSampleRecord, SourceSliceState
from training.big_vae.source_pool import _consume_slice_from_source_state, _make_source_slice_state, _max_source_state_shape_capacity, _prune_exhausted_source_states, _remaining_source_state_slices, _source_state_slice_shape, _source_states_total_remaining_slices, _source_states_uniqueness_keys
from training.big_vae.source_sampling import _fetch_source_samples, _source_sample_record_to_device, _source_sample_uniqueness_key



def _compute_consumed_batch_source_diversity_stats(
    source_states: Sequence[SourceSliceState],
    *,
    used_source_indices: Sequence[int],
    source_pool_remaining_slices_pre: int,
    source_pool_remaining_slices_post: int,
) -> dict[str, float]:
    stats: dict[str, float] = {
        "source_pool_size": 0.0,
        "source_pool_unique_named_models": 0.0,
        "source_pool_missing_model_names": 0.0,
        "source_pool_remaining_slices_pre": float(source_pool_remaining_slices_pre),
        "source_pool_remaining_slices_post": float(source_pool_remaining_slices_post),
        "batch_sources_used": 0.0,
        "batch_unique_models": 0.0,
        "batch_unique_named_models": 0.0,
        "batch_missing_model_sources": 0.0,
        "batch_model_entropy": 0.0,
        "batch_model_perplexity": 0.0,
    }
    if not source_states:
        return stats

    pool_named_models = {
        str(state.source.model_name).strip()
        for state in source_states
        if str(state.source.model_name).strip()
    }
    pool_missing_model_names = sum(
        1
        for state in source_states
        if not str(state.source.model_name).strip()
    )
    stats["source_pool_size"] = float(len(source_states))
    stats["source_pool_unique_named_models"] = float(len(pool_named_models))
    stats["source_pool_missing_model_names"] = float(pool_missing_model_names)

    if not used_source_indices:
        return stats

    source_counts: dict[int, int] = {}
    for source_idx in used_source_indices:
        source_counts[int(source_idx)] = source_counts.get(int(source_idx), 0) + 1

    batch_named_models: set[str] = set()
    batch_missing_model_sources = 0
    effective_model_counts: dict[str, int] = {}
    for source_idx, count in source_counts.items():
        model_name = str(source_states[source_idx].source.model_name).strip()
        if model_name:
            batch_named_models.add(model_name)
            label = model_name
        else:
            batch_missing_model_sources += 1
            label = f"__unknown_source_{source_idx}"
        effective_model_counts[label] = effective_model_counts.get(label, 0) + int(count)

    entropy = 0.0
    total_items = sum(effective_model_counts.values())
    if total_items > 0:
        for count in effective_model_counts.values():
            prob = float(count) / float(total_items)
            entropy -= prob * math.log(max(prob, 1e-12))

    stats["batch_sources_used"] = float(len(source_counts))
    stats["batch_unique_models"] = float(len(effective_model_counts))
    stats["batch_unique_named_models"] = float(len(batch_named_models))
    stats["batch_missing_model_sources"] = float(batch_missing_model_sources)
    stats["batch_model_entropy"] = float(entropy)
    stats["batch_model_perplexity"] = float(math.exp(entropy)) if total_items > 0 else 0.0
    return stats



def _build_training_batch_from_source_states(
    source_states: Sequence[SourceSliceState],
    *,
    batch_size: int,
    start_offset: int = 0,
    target_x_rows: int | None = None,
    target_d_in: int | None = None,
    target_d_out: int | None = None,
) -> ConsumedSourceBatch:
    if not source_states:
        raise ValueError("source_states must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    source_pool_remaining_slices_pre = _source_states_total_remaining_slices(source_states)
    if source_pool_remaining_slices_pre < batch_size:
        raise ValueError(
            "source pool does not have enough remaining slices to build a batch without replacement: "
            f"needed={batch_size} available={source_pool_remaining_slices_pre}"
        )

    source_pool_unique_named_models = len(
        {
            str(state.source.model_name).strip()
            for state in source_states
            if str(state.source.model_name).strip()
        }
    )
    source_pool_missing_model_names = sum(
        1
        for state in source_states
        if not str(state.source.model_name).strip()
    )

    raw_slices: list[tuple[torch.Tensor, torch.Tensor]] = []
    used_source_indices: list[int] = []
    cursor = int(start_offset) % len(source_states)
    stagnant_scans = 0

    while len(raw_slices) < batch_size:
        state = source_states[cursor]
        if _remaining_source_state_slices(state) <= 0:
            stagnant_scans += 1
            if stagnant_scans >= len(source_states):
                raise RuntimeError("source pool stalled while assembling a batch without replacement despite precomputed capacity")
            cursor = (cursor + 1) % len(source_states)
            continue

        sliced = _consume_slice_from_source_state(state)
        if sliced is None:
            stagnant_scans += 1
            if stagnant_scans >= len(source_states):
                raise RuntimeError(
                    "source pool stalled while assembling a batch without replacement despite precomputed capacity"
                )
            cursor = (cursor + 1) % len(source_states)
            continue

        stagnant_scans = 0
        W_i, x_i = sliced
        raw_slices.append((W_i, x_i))
        used_source_indices.append(cursor)
        cursor = (cursor + 1) % len(source_states)

    max_x_rows = max(int(x_i.shape[0]) for _, x_i in raw_slices)
    max_d_in = max(int(W_i.shape[0]) for W_i, _ in raw_slices)
    max_d_out = max(int(W_i.shape[1]) for W_i, _ in raw_slices)
    if target_x_rows is not None:
        max_x_rows = int(target_x_rows)
    if target_d_in is not None:
        max_d_in = int(target_d_in)
    if target_d_out is not None:
        max_d_out = int(target_d_out)
    W_batch, x_batch, x_mask_batch, d_in_mask_batch, d_out_mask_batch = _materialize_padded_slice_batch(
        raw_slices,
        target_x_rows=max_x_rows,
        target_d_in=max_d_in,
        target_d_out=max_d_out,
    )

    return ConsumedSourceBatch(
        W=W_batch,
        x=x_batch,
        x_mask=x_mask_batch,
        d_in_mask=d_in_mask_batch,
        d_out_mask=d_out_mask_batch,
        used_source_indices=tuple(used_source_indices),
        next_start_offset=int(cursor),
        source_pool_size=int(len(source_states)),
        source_pool_unique_named_models=int(source_pool_unique_named_models),
        source_pool_missing_model_names=int(source_pool_missing_model_names),
        source_pool_remaining_slices_pre=int(source_pool_remaining_slices_pre),
        source_pool_remaining_slices_post=int(_source_states_total_remaining_slices(source_states)),
    )



def _build_training_batch_from_source_samples(
    source_samples: Sequence[SourceSampleRecord],
    *,
    max_T_patches: int,
    max_d_out: int,
    patch_size: int,
    batch_size: int,
    start_offset: int = 0,
    target_x_rows: int | None = None,
    target_d_in: int | None = None,
    target_d_out: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not source_samples:
        raise ValueError("source_samples must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    max_x_rows = max(int(record.x.shape[0]) for record in source_samples)
    if target_x_rows is not None:
        max_x_rows = int(target_x_rows)
    normalized_sources = [
        (
            *_pad_x_rows_with_mask(record.x, max_x_rows),
            record.W,
        )
        for record in source_samples
    ]

    assignment = _round_robin_source_indices(
        num_sources=len(normalized_sources),
        batch_size=batch_size,
        start_offset=start_offset,
    )
    counts = [0] * len(normalized_sources)
    for source_idx in assignment:
        counts[source_idx] += 1

    source_batches: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    max_d_in = 0
    max_batch_d_out = 0
    for source_idx, count in enumerate(counts):
        if count <= 0:
            continue
        x_src, x_mask_src, W_src = normalized_sources[source_idx]
        W_part, x_part = _slice_sample(
            W_src,
            x_src,
            max_T_patches=max_T_patches,
            max_d_out=max_d_out,
            patch_size=patch_size,
            batch_size=count,
        )
        max_d_in = max(max_d_in, int(W_part.shape[1]))
        max_batch_d_out = max(max_batch_d_out, int(W_part.shape[2]))
        x_mask_part = x_mask_src.unsqueeze(0).expand(count, -1)
        source_batches[source_idx] = (W_part, x_part, x_mask_part)

    if target_d_in is not None:
        max_d_in = int(target_d_in)
    if target_d_out is not None:
        max_batch_d_out = int(target_d_out)

    source_offsets = [0] * len(normalized_sources)
    first_batch = next(iter(source_batches.values()), None)
    if first_batch is None:
        raise RuntimeError("source_batches must not be empty")
    first_W_batch, first_x_batch, _first_x_mask_batch = first_batch
    W_batch = first_W_batch.new_zeros((int(batch_size), int(max_d_in), int(max_batch_d_out)))
    x_batch = first_x_batch.new_zeros((int(batch_size), int(max_x_rows), int(max_d_in)))
    x_mask_batch = torch.zeros((int(batch_size), int(max_x_rows)), device=first_x_batch.device, dtype=torch.bool)
    d_in_mask_batch = torch.zeros((int(batch_size), int(max_d_in)), device=first_W_batch.device, dtype=torch.bool)
    d_out_mask_batch = torch.zeros((int(batch_size), int(max_batch_d_out)), device=first_W_batch.device, dtype=torch.bool)
    for batch_idx, source_idx in enumerate(assignment):
        W_part, x_part, x_mask_part = source_batches[source_idx]
        cursor = source_offsets[source_idx]
        source_offsets[source_idx] += 1
        W_i = W_part[cursor]
        x_i = x_part[cursor]
        x_mask_i = x_mask_part[cursor]
        d_in_i = int(W_i.shape[0])
        d_out_i = int(W_i.shape[1])
        W_batch[batch_idx, :d_in_i, :d_out_i] = W_i
        x_batch[batch_idx, :, :d_in_i] = x_i
        x_mask_batch[batch_idx] = x_mask_i
        d_in_mask_batch[batch_idx, :d_in_i] = True
        d_out_mask_batch[batch_idx, :d_out_i] = True

    return (
        W_batch,
        x_batch,
        x_mask_batch,
        d_in_mask_batch,
        d_out_mask_batch,
    )



def _ensure_source_state_pool_capacity(
    *,
    current_source_states: list[SourceSliceState] | None,
    required_remaining_slices: int,
    target_source_pool_size: int,
    max_active_source_pool_size: int,
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[Any] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
    uniqueness: str,
    deferred_samples: deque[SourceSampleRecord] | None,
    max_deferred_samples: int,
    refill_fetch_batch_size: int,
    min_d_in: int,
    min_d_out: int,
    max_T_patches: int,
    curriculum_max_d_out: int,
    patch_size: int,
    synthetic_layer_enabled: bool,
    synthetic_n_rows: int,
    synthetic_d_in: int,
    synthetic_d_out: int,
    synthetic_x_std: float,
    synthetic_w_std: float,
) -> list[SourceSliceState]:
    source_states = _prune_exhausted_source_states(current_source_states)
    if required_remaining_slices <= 0:
        return source_states

    refill_attempts = 0
    max_refill_attempts = max(int(required_remaining_slices) * 8, int(target_source_pool_size) * 8, 32)
    while True:
        remaining_slices = _source_states_total_remaining_slices(source_states)
        if remaining_slices >= required_remaining_slices and len(source_states) >= target_source_pool_size:
            break
        if len(source_states) >= max_active_source_pool_size:
            break
        if refill_attempts >= max_refill_attempts and remaining_slices >= required_remaining_slices:
            break

        existing_uniqueness_keys = _source_states_uniqueness_keys(source_states, uniqueness=uniqueness)
        if synthetic_layer_enabled:
            room = max(1, int(max_active_source_pool_size) - len(source_states))
            effective_fetch_batch_size = max(1, min(int(refill_fetch_batch_size), room))
            fetched_records = []
            next_source_idx = len(source_states)
            for sample_offset in range(effective_fetch_batch_size):
                x_syn, W_syn = _sample_synthetic_layer(
                    device=torch.device("cpu"),
                    n_rows=synthetic_n_rows,
                    d_in=synthetic_d_in,
                    d_out=synthetic_d_out,
                    x_std=synthetic_x_std,
                    w_std=synthetic_w_std,
                    max_x_rows=max_x_rows,
                )
                fetched_records.append(
                    SourceSampleRecord(
                        x=x_syn,
                        W=W_syn,
                        model_name=f"synthetic_{next_source_idx + sample_offset}",
                    )
                )
        else:
            room = max(1, int(max_active_source_pool_size) - len(source_states))
            target_gap = max(1, int(target_source_pool_size) - len(source_states))
            effective_fetch_batch_size = max(1, min(int(refill_fetch_batch_size), room, target_gap))
            fetched_records = _fetch_source_samples(
                rank=rank,
                device=device,
                dataset_iter=dataset_iter,
                use_broadcast=use_broadcast,
                max_x_rows=max_x_rows,
                logger=logger,
                num_samples=effective_fetch_batch_size,
                uniqueness=uniqueness,
                deferred_samples=deferred_samples,
                max_deferred_samples=max_deferred_samples,
                min_d_in=min_d_in,
                min_d_out=min_d_out,
                existing_uniqueness_keys=existing_uniqueness_keys,
            )

        progress_made = False
        for record in fetched_records:
            source_states.append(
                _make_source_slice_state(
                    record,
                    max_T_patches=max_T_patches,
                    max_d_out=curriculum_max_d_out,
                    patch_size=patch_size,
                )
            )
            progress_made = True
            if len(source_states) >= max_active_source_pool_size:
                break

        source_states = _prune_exhausted_source_states(source_states)
        refill_attempts += 1
        if not progress_made and refill_attempts >= max_refill_attempts:
            break

    return source_states


__all__ = [
    '_compute_consumed_batch_source_diversity_stats',
    '_build_training_batch_from_source_states',
    '_build_training_batch_from_source_samples',
    '_ensure_source_state_pool_capacity',
]
