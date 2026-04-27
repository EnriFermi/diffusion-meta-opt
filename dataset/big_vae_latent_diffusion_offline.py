from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from dataset.big_vae_offline import OfflineBigVAEDataset, infer_layer_depth, infer_layer_type
from dataset.shared.types import SharedSample
from training.big_vae_latent_diffusion import (
    build_cond_global_from_dist_var_pooled,
    encode_big_vae_layer_batch,
    latent_diffusion_layer_type_to_id,
)


OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION = 1
_DISTRIBUTION_MATCH_GROUP_KEY_ALIASES = {
    "dataset": "dataset",
    "model": "model",
    "layer_type": "layer_type",
    "layer": "layer",
    "depth": "depth",
    "shape": "shape",
    "source": "source",
    "d_in_bucket": "d_in_bucket",
    "d_out_bucket": "d_out_bucket",
    "num_params_bucket": "num_params_bucket",
}


@dataclass(slots=True)
class _ExplicitSliceTarget:
    patch_size: int
    target_T_patches: int
    target_d_out: int

    @property
    def target_d_in(self) -> int:
        return int(self.patch_size) * int(self.target_T_patches)


@dataclass(slots=True)
class _LatentDiffusionSourceState:
    source: SharedSample
    patch_size: int
    row_patch_groups: tuple[torch.Tensor, ...] | None
    col_groups: tuple[torch.Tensor, ...] | None
    row_cursor: int = 0
    col_cursor: int = 0
    single_unsliced_consumed: bool = False


@dataclass(slots=True)
class _DistributionMatchConfig:
    mode: str
    group_keys: tuple[str, ...]
    max_latent_records: int
    max_slices_per_source_record: int


def _source_key(model_name: str, layer_name: str) -> str:
    payload = f"{str(model_name).strip()}\n{str(layer_name).strip()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _prepare_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        return tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _directory_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return total


def _resolve_explicit_slice_target(builder_cfg: Mapping[str, Any], *, patch_size: int) -> _ExplicitSliceTarget:
    target_T_patches = int(builder_cfg.get("target_T_patches", 0))
    target_d_out = int(builder_cfg.get("target_d_out", 0))
    if target_T_patches <= 0:
        raise ValueError("latent_diffusion_dataset.builder.target_T_patches must be > 0")
    if target_d_out <= 0:
        raise ValueError("latent_diffusion_dataset.builder.target_d_out must be > 0")
    return _ExplicitSliceTarget(
        patch_size=int(patch_size),
        target_T_patches=target_T_patches,
        target_d_out=target_d_out,
    )


def _normalize_distribution_match_mode(value: Any) -> str:
    normalized = str(value or "source_exhaustive").strip().lower()
    aliases = {
        "": "source_exhaustive",
        "none": "source_exhaustive",
        "off": "source_exhaustive",
        "source_exhaustive": "source_exhaustive",
        "train_style_without_replacement": "source_exhaustive",
        "exhaustive": "source_exhaustive",
        "source_proportional_joint": "source_proportional_joint",
        "joint": "source_proportional_joint",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"source_exhaustive", "source_proportional_joint"}:
        raise ValueError(f"Unsupported latent_diffusion_dataset.builder.distribution_match.mode: {value!r}")
    return resolved


def _normalize_distribution_match_group_keys(value: Any) -> tuple[str, ...]:
    if value is None:
        raw_items: list[str] = []
    elif isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        raw_items = [str(item).strip() for item in value]
    else:
        raise TypeError("latent_diffusion_dataset.builder.distribution_match.group_keys must be a string or sequence")

    normalized: list[str] = []
    for item in raw_items:
        if not item:
            continue
        key = _DISTRIBUTION_MATCH_GROUP_KEY_ALIASES.get(item.strip().lower())
        if key is None:
            valid = ", ".join(sorted(_DISTRIBUTION_MATCH_GROUP_KEY_ALIASES))
            raise ValueError(
                f"Unsupported latent_diffusion distribution_match.group_keys item {item!r}; valid keys: {valid}"
            )
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        return ("dataset", "model", "layer_type", "layer")
    return tuple(normalized)


def _resolve_distribution_match_cfg(builder_cfg: Mapping[str, Any]) -> _DistributionMatchConfig:
    raw_cfg = builder_cfg.get("distribution_match", {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, Mapping):
        raise TypeError("latent_diffusion_dataset.builder.distribution_match must be a mapping")
    max_latent_records = max(0, int(builder_cfg.get("max_latent_records", 0)))
    max_slices_per_source_record = max(1, int(builder_cfg.get("max_slices_per_source_record", 1)))
    return _DistributionMatchConfig(
        mode=_normalize_distribution_match_mode(raw_cfg.get("mode", "source_exhaustive")),
        group_keys=_normalize_distribution_match_group_keys(
            raw_cfg.get("group_keys", ("dataset", "model", "layer_type", "layer"))
        ),
        max_latent_records=max_latent_records,
        max_slices_per_source_record=max_slices_per_source_record,
    )


def _pow2_bucket(value: int) -> str:
    value = int(value)
    if value <= 0:
        return "0"
    lower = 1 << max(0, int(value).bit_length() - 1)
    upper = max(lower, (lower << 1) - 1)
    return f"{lower}-{upper}"


def _shape_key(d_in: int, d_out: int) -> str:
    return f"{int(d_in)}x{int(d_out)}"


def _sample_dataset_names(meta: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(meta, Mapping):
        return []
    image_meta = meta.get("image_meta", [])
    if not isinstance(image_meta, list):
        return []
    dataset_names = {
        str(item.get("dataset_name")).strip()
        for item in image_meta
        if isinstance(item, Mapping) and str(item.get("dataset_name", "")).strip()
    }
    return sorted(dataset_names)


def _primary_dataset_name(meta: Mapping[str, Any] | None) -> str:
    dataset_names = _sample_dataset_names(meta)
    return dataset_names[0] if dataset_names else "<unknown_dataset>"


def _latent_diffusion_source_slice_capacity_from_shape(
    *,
    d_in: int,
    d_out: int,
    target: _ExplicitSliceTarget,
) -> int:
    total_patches = int(d_in) // int(target.patch_size)
    if total_patches < int(target.target_T_patches) or int(d_out) < int(target.target_d_out):
        return 0

    row_groups = 1 if total_patches <= int(target.target_T_patches) else total_patches // int(target.target_T_patches)
    col_groups = 1 if int(d_out) <= int(target.target_d_out) else int(d_out) // int(target.target_d_out)
    return max(1, min(int(row_groups), int(col_groups)))


def _distribution_match_group_key_for_sample(
    sample: SharedSample,
    *,
    group_keys: Sequence[str],
) -> tuple[str, ...]:
    model_name = str(sample.model_name).strip() or "<unknown_model>"
    layer_name = str(sample.layer_name).strip() or "<unknown_layer>"
    meta = dict(sample.meta or {})
    d_in = int(sample.weight.shape[0])
    d_out = int(sample.weight.shape[1])
    dataset_name = _primary_dataset_name(meta)
    layer_type = infer_layer_type(layer_name)
    depth_value = infer_layer_depth(layer_name)
    depth_label = str(depth_value) if depth_value is not None else "unknown"
    source = _source_key(model_name, layer_name)

    components: list[str] = []
    for key in group_keys:
        if key == "dataset":
            components.append(dataset_name)
        elif key == "model":
            components.append(model_name)
        elif key == "layer_type":
            components.append(layer_type)
        elif key == "layer":
            components.append(layer_name)
        elif key == "depth":
            components.append(depth_label)
        elif key == "shape":
            components.append(_shape_key(d_in, d_out))
        elif key == "source":
            components.append(source)
        elif key == "d_in_bucket":
            components.append(_pow2_bucket(d_in))
        elif key == "d_out_bucket":
            components.append(_pow2_bucket(d_out))
        elif key == "num_params_bucket":
            components.append(_pow2_bucket(d_in * d_out))
        else:
            raise ValueError(f"Unsupported latent diffusion distribution_match group key: {key!r}")
    return tuple(components)


def _distribution_match_source_info(
    sample: SharedSample,
    *,
    target: _ExplicitSliceTarget,
    group_keys: Sequence[str],
    max_slices_per_source_record: int,
) -> tuple[tuple[str, ...], int] | None:
    d_in = int(sample.weight.shape[0])
    d_out = int(sample.weight.shape[1])
    slice_capacity = _latent_diffusion_source_slice_capacity_from_shape(d_in=d_in, d_out=d_out, target=target)
    if slice_capacity <= 0:
        return None
    effective_capacity = min(int(slice_capacity), int(max_slices_per_source_record))
    return _distribution_match_group_key_for_sample(sample, group_keys=group_keys), int(effective_capacity)


def _allocate_capped_proportional_quotas(
    *,
    weights: Mapping[tuple[str, ...], int],
    capacities: Mapping[tuple[str, ...], int],
    total_target: int,
) -> dict[tuple[str, ...], int]:
    quotas = {group_key: 0 for group_key in weights.keys()}
    remaining_capacity = {
        group_key: max(0, int(capacities.get(group_key, 0)))
        for group_key in weights.keys()
    }
    remaining_target = min(max(0, int(total_target)), sum(remaining_capacity.values()))
    order = {group_key: idx for idx, group_key in enumerate(weights.keys())}
    active = [group_key for group_key in weights.keys() if remaining_capacity[group_key] > 0]

    while remaining_target > 0 and active:
        total_weight = sum(max(0, int(weights[group_key])) for group_key in active)
        if total_weight <= 0:
            for group_key in active:
                if remaining_target <= 0:
                    break
                take_n = min(remaining_capacity[group_key], remaining_target)
                quotas[group_key] += int(take_n)
                remaining_capacity[group_key] -= int(take_n)
                remaining_target -= int(take_n)
            break

        saturated: list[tuple[str, ...]] = []
        floor_allocs: dict[tuple[str, ...], int] = {}
        remainders: list[tuple[float, int, tuple[str, ...]]] = []
        for group_key in active:
            cap = int(remaining_capacity[group_key])
            ideal = float(remaining_target) * float(max(0, int(weights[group_key]))) / float(total_weight)
            if ideal >= float(cap):
                quotas[group_key] += int(cap)
                remaining_target -= int(cap)
                remaining_capacity[group_key] = 0
                saturated.append(group_key)
                continue
            base = min(cap, int(ideal))
            floor_allocs[group_key] = int(base)
            remainders.append((ideal - float(base), int(order[group_key]), group_key))

        if saturated:
            active = [group_key for group_key in active if remaining_capacity[group_key] > 0]
            continue

        base_sum = 0
        for group_key, base in floor_allocs.items():
            quotas[group_key] += int(base)
            remaining_capacity[group_key] -= int(base)
            base_sum += int(base)
        remaining_target -= int(base_sum)

        remainders.sort(key=lambda item: (-item[0], item[1]))
        for _remainder, _order_idx, group_key in remainders:
            if remaining_target <= 0:
                break
            if int(remaining_capacity[group_key]) <= 0:
                continue
            quotas[group_key] += 1
            remaining_capacity[group_key] -= 1
            remaining_target -= 1
        break

    return quotas


def _sample_slot_count_without_replacement(
    *,
    num_slots: int,
    success_remaining: int,
    population_remaining: int,
    rng: random.Random,
) -> int:
    slots = max(0, int(num_slots))
    successes_left = max(0, int(success_remaining))
    population_left = max(0, int(population_remaining))
    if slots <= 0 or successes_left <= 0 or population_left <= 0:
        return 0

    taken = 0
    for _ in range(min(slots, population_left)):
        if successes_left <= 0 or population_left <= 0:
            break
        prob = float(successes_left) / float(population_left)
        if rng.random() < prob:
            taken += 1
            successes_left -= 1
        population_left -= 1
    return int(taken)


def _build_without_replacement_index_groups(
    *,
    num_items: int,
    group_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> tuple[torch.Tensor, ...] | None:
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if group_size >= num_items:
        return None

    num_groups = num_items // group_size
    order = torch.randperm(num_items, generator=rng, device=device)
    groups: list[torch.Tensor] = []
    for group_idx in range(num_groups):
        group = order[group_idx * group_size : (group_idx + 1) * group_size].sort().values
        groups.append(group)
    return tuple(groups)


def _make_latent_diffusion_source_state(
    sample: SharedSample,
    *,
    target: _ExplicitSliceTarget,
    rng: torch.Generator,
) -> _LatentDiffusionSourceState | None:
    x_cpu = _prepare_cpu_tensor(sample.x)
    weight_cpu = _prepare_cpu_tensor(sample.weight)
    d_in, d_out = int(weight_cpu.shape[0]), int(weight_cpu.shape[1])
    available_full_patches = d_in // int(target.patch_size)
    if available_full_patches < int(target.target_T_patches) or d_out < int(target.target_d_out):
        return None

    meta = dict(sample.meta or {})
    meta["latent_diffusion_builder_explicit_slice"] = False
    meta["latent_diffusion_builder_slice_mode"] = "train_style_without_replacement"
    meta["latent_diffusion_target_T_patches"] = int(target.target_T_patches)
    meta["latent_diffusion_target_d_in"] = int(target.target_d_in)
    meta["latent_diffusion_target_d_out"] = int(target.target_d_out)
    meta["latent_diffusion_source_d_in"] = d_in
    meta["latent_diffusion_source_d_out"] = d_out
    meta["latent_diffusion_available_full_patches"] = int(available_full_patches)
    return _LatentDiffusionSourceState(
        source=SharedSample(
            model_name=str(sample.model_name),
            layer_name=str(sample.layer_name),
            weight=weight_cpu,
            x=x_cpu,
            y=sample.y,
            meta=meta,
        ),
        patch_size=int(target.patch_size),
        row_patch_groups=_build_without_replacement_index_groups(
            num_items=int(available_full_patches),
            group_size=int(target.target_T_patches),
            device=weight_cpu.device,
            rng=rng,
        ),
        col_groups=_build_without_replacement_index_groups(
            num_items=int(d_out),
            group_size=int(target.target_d_out),
            device=weight_cpu.device,
            rng=rng,
        ),
    )


def _remaining_latent_diffusion_source_slices(state: _LatentDiffusionSourceState) -> int:
    capacities: list[int] = []
    if state.row_patch_groups is not None:
        capacities.append(len(state.row_patch_groups) - int(state.row_cursor))
    if state.col_groups is not None:
        capacities.append(len(state.col_groups) - int(state.col_cursor))
    if not capacities:
        return 0 if state.single_unsliced_consumed else 1
    return max(0, min(capacities))


def _prune_exhausted_latent_diffusion_source_states_with_offset(
    source_states: list[_LatentDiffusionSourceState] | None,
    start_offset: int,
) -> tuple[list[_LatentDiffusionSourceState], int]:
    if not source_states:
        return [], 0

    survivors: list[_LatentDiffusionSourceState] = []
    remapped_offset = 0
    normalized_offset = int(start_offset)
    for idx, state in enumerate(source_states):
        if _remaining_latent_diffusion_source_slices(state) <= 0:
            continue
        if idx < normalized_offset:
            remapped_offset += 1
        survivors.append(state)

    if not survivors:
        return [], 0
    return survivors, remapped_offset % len(survivors)


def _latent_diffusion_source_states_total_remaining_slices(
    source_states: Sequence[_LatentDiffusionSourceState] | None,
) -> int:
    if not source_states:
        return 0
    return sum(_remaining_latent_diffusion_source_slices(state) for state in source_states)


def _consume_slice_from_latent_diffusion_source_state(
    state: _LatentDiffusionSourceState,
) -> SharedSample | None:
    remaining = _remaining_latent_diffusion_source_slices(state)
    if remaining <= 0:
        return None

    W_i = state.source.weight
    x_i = state.source.x
    d_in = int(W_i.shape[0])
    d_out = int(W_i.shape[1])
    total_patches = d_in // int(state.patch_size)
    meta = dict(state.source.meta or {})

    if state.row_patch_groups is not None:
        patch_idx = state.row_patch_groups[state.row_cursor]
        state.row_cursor += 1
        offsets = torch.arange(state.patch_size, device=W_i.device)
        row_idx = (patch_idx.unsqueeze(1) * state.patch_size + offsets.unsqueeze(0)).flatten()
        row_idx = row_idx.clamp(max=d_in - 1)
        W_i = W_i.index_select(dim=0, index=row_idx)
        x_i = x_i.index_select(dim=1, index=row_idx)
        meta["latent_diffusion_builder_explicit_slice"] = True
        meta["latent_diffusion_row_patch_idx"] = [int(idx) for idx in patch_idx.tolist()]
        meta["latent_diffusion_row_idx_preview"] = [int(idx) for idx in row_idx[:64].tolist()]
    else:
        meta["latent_diffusion_row_patch_idx"] = [int(idx) for idx in range(total_patches)]

    if state.col_groups is not None:
        col_idx = state.col_groups[state.col_cursor]
        state.col_cursor += 1
        W_i = W_i.index_select(dim=1, index=col_idx)
        meta["latent_diffusion_builder_explicit_slice"] = True
        meta["latent_diffusion_col_idx"] = [int(idx) for idx in col_idx.tolist()]
    else:
        meta["latent_diffusion_col_idx"] = [int(idx) for idx in range(d_out)]

    if state.row_patch_groups is None and state.col_groups is None:
        state.single_unsliced_consumed = True

    return SharedSample(
        model_name=str(state.source.model_name),
        layer_name=str(state.source.layer_name),
        weight=W_i.contiguous(),
        x=x_i.contiguous(),
        y=state.source.y,
        meta=meta,
    )


def _build_latent_diffusion_batch_from_source_states(
    source_states: Sequence[_LatentDiffusionSourceState],
    *,
    batch_size: int,
    start_offset: int = 0,
) -> tuple[dict[str, Any], int, tuple[int, ...]]:
    if not source_states:
        raise ValueError("source_states must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    source_pool_remaining_slices_pre = _latent_diffusion_source_states_total_remaining_slices(source_states)
    if source_pool_remaining_slices_pre < batch_size:
        raise ValueError(
            "source pool does not have enough remaining slices to build a batch without replacement: "
            f"needed={batch_size} available={source_pool_remaining_slices_pre}"
        )

    sliced_samples: list[SharedSample] = []
    used_source_indices: list[int] = []
    cursor = int(start_offset) % len(source_states)
    stagnant_scans = 0

    while len(sliced_samples) < batch_size:
        state = source_states[cursor]
        if _remaining_latent_diffusion_source_slices(state) <= 0:
            stagnant_scans += 1
            if stagnant_scans >= len(source_states):
                raise RuntimeError(
                    "source pool stalled while assembling latent diffusion slices without replacement"
                )
            cursor = (cursor + 1) % len(source_states)
            continue

        sliced = _consume_slice_from_latent_diffusion_source_state(state)
        if sliced is None:
            stagnant_scans += 1
            if stagnant_scans >= len(source_states):
                raise RuntimeError(
                    "source pool stalled while assembling latent diffusion slices without replacement"
                )
            cursor = (cursor + 1) % len(source_states)
            continue

        stagnant_scans = 0
        sliced_samples.append(sliced)
        used_source_indices.append(cursor)
        cursor = (cursor + 1) % len(source_states)

    return _pad_source_samples(sliced_samples), int(cursor), tuple(used_source_indices)


class BigVAELatentDiffusionOfflineWriter:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        overwrite_existing: bool,
        chunk_size_records: int,
        patch_size: int,
        target_T_patches: int,
        target_d_out: int,
        store_decoder_aux_tensors: bool = False,
        logger: logging.Logger | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.root_dir = Path(str(root_dir))
        if not str(self.root_dir).strip():
            raise ValueError("latent diffusion dataset root_dir must be non-empty")
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.chunk_size_records = max(1, int(chunk_size_records))
        self.patch_size = max(1, int(patch_size))
        self.target_T_patches = max(1, int(target_T_patches))
        self.target_d_out = max(1, int(target_d_out))
        self.store_decoder_aux_tensors = bool(store_decoder_aux_tensors)
        self.config_snapshot = dict(config_snapshot or {})
        self.chunks_dir = self.root_dir / "chunks"
        self.manifest_path = self.root_dir / "manifest.json"
        self.stats_path = self.root_dir / "latent_stats.pt"
        self._chunk_buffer: list[dict[str, Any]] = []
        self._chunk_index = 0
        self._accepted_records = 0
        self._closed = False
        self._z_dim: int | None = None
        self._cond_dim: int | None = None
        self._cond_global_dim: int | None = None
        self._latent_sum: torch.Tensor | None = None
        self._latent_sumsq: torch.Tensor | None = None
        self._prepare_root(overwrite_existing=bool(overwrite_existing))

    def ingest(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("writer is already closed")
        latent_mu = record.get("latent_mu")
        cond_patch = record.get("cond_patch")
        cond_global = record.get("cond_global")
        patch_mask = record.get("patch_mask")
        if not torch.is_tensor(latent_mu) or latent_mu.ndim != 1:
            raise TypeError(f"record.latent_mu must be rank-1 tensor, got {type(latent_mu)!r}")
        if not torch.is_tensor(cond_patch) or cond_patch.ndim != 2:
            raise TypeError(f"record.cond_patch must be rank-2 tensor, got {type(cond_patch)!r}")
        if not torch.is_tensor(patch_mask) or patch_mask.ndim != 1:
            raise TypeError(f"record.patch_mask must be rank-1 tensor, got {type(patch_mask)!r}")
        if int(cond_patch.shape[0]) != int(patch_mask.shape[0]):
            raise ValueError(
                f"record.cond_patch and record.patch_mask must align on T, got {tuple(cond_patch.shape)} and {tuple(patch_mask.shape)}"
            )

        latent_mu_cpu = _prepare_cpu_tensor(latent_mu)
        cond_patch_cpu = _prepare_cpu_tensor(cond_patch)
        cond_global_cpu: torch.Tensor | None = None
        if cond_global is not None:
            if not torch.is_tensor(cond_global) or cond_global.ndim != 1:
                raise TypeError(f"record.cond_global must be rank-1 tensor when provided, got {type(cond_global)!r}")
            cond_global_cpu = _prepare_cpu_tensor(cond_global)
        patch_mask_cpu = patch_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        target_X = record.get("X")
        target_W = record.get("W")
        target_x_mask = record.get("x_mask")
        target_d_in_mask = record.get("d_in_mask")
        target_d_out_mask = record.get("d_out_mask")

        z_dim = int(latent_mu_cpu.numel())
        cond_dim = int(cond_patch_cpu.shape[1])
        cond_global_dim = int(cond_global_cpu.numel()) if cond_global_cpu is not None else 0
        if self._z_dim is None:
            self._z_dim = z_dim
            self._latent_sum = torch.zeros(z_dim, dtype=torch.float64)
            self._latent_sumsq = torch.zeros(z_dim, dtype=torch.float64)
        if self._cond_dim is None:
            self._cond_dim = cond_dim
        if self._cond_global_dim is None:
            self._cond_global_dim = cond_global_dim
        if z_dim != int(self._z_dim):
            raise ValueError(f"inconsistent z_dim: expected {self._z_dim}, got {z_dim}")
        if cond_dim != int(self._cond_dim):
            raise ValueError(f"inconsistent cond_dim: expected {self._cond_dim}, got {cond_dim}")
        if cond_global_dim != int(self._cond_global_dim):
            raise ValueError(f"inconsistent cond_global_dim: expected {self._cond_global_dim}, got {cond_global_dim}")

        assert self._latent_sum is not None
        assert self._latent_sumsq is not None
        latent_mu_f64 = latent_mu_cpu.to(dtype=torch.float64)
        self._latent_sum += latent_mu_f64
        self._latent_sumsq += latent_mu_f64.pow(2)

        payload = dict(record)
        payload["latent_mu"] = latent_mu_cpu
        payload["cond_patch"] = cond_patch_cpu
        if cond_global_cpu is not None:
            payload["cond_global"] = cond_global_cpu
        payload["patch_mask"] = patch_mask_cpu
        latent_logvar = payload.get("latent_logvar")
        if torch.is_tensor(latent_logvar):
            payload["latent_logvar"] = _prepare_cpu_tensor(latent_logvar)
        if self.store_decoder_aux_tensors:
            if not torch.is_tensor(target_X) or target_X.ndim != 2:
                raise TypeError("record.X must be rank-2 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_W) or target_W.ndim != 2:
                raise TypeError("record.W must be rank-2 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_x_mask) or target_x_mask.ndim != 1:
                raise TypeError("record.x_mask must be rank-1 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_d_in_mask) or target_d_in_mask.ndim != 1:
                raise TypeError("record.d_in_mask must be rank-1 tensor when store_decoder_aux_tensors=true")
            if not torch.is_tensor(target_d_out_mask) or target_d_out_mask.ndim != 1:
                raise TypeError("record.d_out_mask must be rank-1 tensor when store_decoder_aux_tensors=true")
            payload["X"] = _prepare_cpu_tensor(target_X)
            payload["W"] = _prepare_cpu_tensor(target_W)
            payload["x_mask"] = target_x_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
            payload["d_in_mask"] = target_d_in_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
            payload["d_out_mask"] = target_d_out_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        self._chunk_buffer.append(payload)
        self._accepted_records += 1
        if len(self._chunk_buffer) >= self.chunk_size_records:
            self._flush_chunk()

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._build_manifest()
        self._flush_chunk()
        self._write_stats()
        _atomic_write_json(self.manifest_path, self._build_manifest())
        self._closed = True
        return self._build_manifest()

    def _prepare_root(self, *, overwrite_existing: bool) -> None:
        if self.root_dir.exists():
            has_payload = any(self.root_dir.iterdir())
            if has_payload and not overwrite_existing:
                raise FileExistsError(
                    f"latent diffusion dataset root already exists and is not empty: {self.root_dir}. "
                    "Set latent_diffusion_dataset.builder.overwrite_existing=true to replace it."
                )
            if has_payload:
                shutil.rmtree(self.root_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def _flush_chunk(self) -> None:
        if not self._chunk_buffer:
            return
        chunk_id = f"latent_chunk_{self._chunk_index:08d}"
        self._chunk_index += 1
        chunk_path = self.chunks_dir / f"{chunk_id}.pt"
        payload = {
            "format_version": OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
            "chunk_id": chunk_id,
            "records": self._chunk_buffer,
            "created_at": float(time.time()),
        }
        torch.save(payload, chunk_path)
        self._chunk_buffer = []

    def _write_stats(self) -> None:
        if self._z_dim is None or self._latent_sum is None or self._latent_sumsq is None or self._accepted_records <= 0:
            raise RuntimeError("cannot finalize latent diffusion dataset stats without accepted records")
        count = max(1, int(self._accepted_records))
        latent_mean = self._latent_sum / float(count)
        latent_var = (self._latent_sumsq / float(count)) - latent_mean.pow(2)
        latent_std = torch.sqrt(latent_var.clamp_min(1e-6))
        torch.save(
            {
                "latent_mean": latent_mean.to(dtype=torch.float32),
                "latent_std": latent_std.to(dtype=torch.float32),
                "count": int(count),
                "z_dim": int(self._z_dim),
                "cond_dim": int(self._cond_dim or 0),
                "cond_global_dim": int(self._cond_global_dim or 0),
            },
            self.stats_path,
        )

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "format_version": OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
            "created_at": float(time.time()),
            "root_dir": str(self.root_dir),
            "accepted_records": int(self._accepted_records),
            "num_chunks": int(self._chunk_index),
            "z_dim": int(self._z_dim or 0),
            "cond_dim": int(self._cond_dim or 0),
            "cond_global_dim": int(self._cond_global_dim or 0),
            "actual_size_bytes": int(_directory_size_bytes(self.root_dir)),
            "actual_size_gb": float(_directory_size_bytes(self.root_dir)) / (1024.0 ** 3),
            "has_decoder_aux_tensors": bool(self.store_decoder_aux_tensors),
            "slice_shape": {
                "patch_size": int(self.patch_size),
                "target_T_patches": int(self.target_T_patches),
                "target_d_in": int(self.patch_size) * int(self.target_T_patches),
                "target_d_out": int(self.target_d_out),
            },
            "layout": {
                "chunks_dir": str(self.chunks_dir.relative_to(self.root_dir)),
                "stats_path": str(self.stats_path.relative_to(self.root_dir)),
            },
            "config_snapshot": self.config_snapshot,
        }


class OfflineBigVAELatentDiffusionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        shuffle_chunks: bool = True,
        shuffle_records_within_chunk: bool = True,
        repeat: bool = True,
        seed: int = 42,
        chunk_cache_size: int = 4,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.stats_path = self.root_dir / "latent_stats.pt"
        self.chunks_dir = self.root_dir / "chunks"
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_records_within_chunk = bool(shuffle_records_within_chunk)
        self.repeat = bool(repeat)
        self.seed = int(seed)
        self.chunk_cache_size = max(1, int(chunk_cache_size))
        self.logger = logging.getLogger(self.__class__.__name__)
        self._chunk_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._record_schema_cache: dict[str, int | bool] | None = None
        self._manifest = self._load_manifest()
        self._stats = self._load_stats()
        self._chunk_paths = sorted(self.chunks_dir.glob("*.pt"))
        if not self._chunk_paths:
            raise FileNotFoundError(f"no latent diffusion chunks found under {self.chunks_dir}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            chunk_indices = list(range(len(self._chunk_paths)))
            if self.shuffle_chunks and len(chunk_indices) > 1:
                rng.shuffle(chunk_indices)
            for chunk_idx in chunk_indices:
                payload = self._load_chunk(chunk_idx)
                records = payload.get("records", [])
                if not isinstance(records, list):
                    raise TypeError(f"chunk {self._chunk_paths[chunk_idx]} has invalid records payload")
                order = list(range(len(records)))
                if self.shuffle_records_within_chunk and len(order) > 1:
                    rng.shuffle(order)
                for record_idx in order:
                    record = records[record_idx]
                    if not isinstance(record, dict):
                        raise TypeError(f"latent diffusion record must be a dict, got {type(record)!r}")
                    yield dict(record)
            if not self.repeat:
                return
            epoch += 1

    def summary(self) -> dict[str, Any]:
        payload = dict(self._manifest)
        record_schema = self._infer_record_schema()
        payload["z_dim"] = max(
            int(payload.get("z_dim", 0)),
            int(self._stats.get("z_dim", 0)),
            int(record_schema.get("z_dim", 0)),
        )
        payload["cond_dim"] = max(
            int(payload.get("cond_dim", 0)),
            int(record_schema.get("cond_dim", 0)),
        )
        payload["cond_global_dim"] = max(
            int(payload.get("cond_global_dim", 0)),
            int(self._stats.get("cond_global_dim", 0)),
            int(record_schema.get("cond_global_dim", 0)),
        )
        payload["has_decoder_aux_tensors"] = bool(
            payload.get("has_decoder_aux_tensors", False) or bool(record_schema.get("has_decoder_aux_tensors", False))
        )
        payload["latent_stats_count"] = int(self._stats.get("count", 0))
        return payload

    def latent_stats(self) -> dict[str, torch.Tensor | int]:
        return {
            "latent_mean": self._stats["latent_mean"].clone(),
            "latent_std": self._stats["latent_std"].clone(),
            "count": int(self._stats.get("count", 0)),
            "cond_global_dim": int(self._stats.get("cond_global_dim", 0)),
        }

    def close(self) -> None:
        self._chunk_cache.clear()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"latent diffusion manifest not found: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"latent diffusion manifest must be a dict, got {type(payload)!r}")
        return payload

    def _load_stats(self) -> dict[str, Any]:
        if not self.stats_path.exists():
            raise FileNotFoundError(f"latent diffusion stats not found: {self.stats_path}")
        payload = torch.load(self.stats_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"latent diffusion stats payload must be a dict, got {type(payload)!r}")
        latent_mean = payload.get("latent_mean")
        latent_std = payload.get("latent_std")
        if not torch.is_tensor(latent_mean) or not torch.is_tensor(latent_std):
            raise TypeError("latent diffusion stats must contain tensors 'latent_mean' and 'latent_std'")
        payload["latent_mean"] = _prepare_cpu_tensor(latent_mean)
        payload["latent_std"] = _prepare_cpu_tensor(latent_std)
        return payload

    def _load_chunk(self, chunk_idx: int) -> dict[str, Any]:
        if chunk_idx in self._chunk_cache:
            payload = self._chunk_cache.pop(chunk_idx)
            self._chunk_cache[chunk_idx] = payload
            return payload
        chunk_path = self._chunk_paths[chunk_idx]
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"latent diffusion chunk payload must be a dict, got {type(payload)!r}: {chunk_path}")
        self._chunk_cache[chunk_idx] = payload
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return payload

    def _infer_record_schema(self) -> dict[str, int | bool]:
        if self._record_schema_cache is not None:
            return dict(self._record_schema_cache)

        inferred = {
            "z_dim": 0,
            "cond_dim": 0,
            "cond_global_dim": 0,
            "has_decoder_aux_tensors": False,
        }
        for chunk_idx in range(len(self._chunk_paths)):
            payload = self._load_chunk(chunk_idx)
            records = payload.get("records", [])
            if not isinstance(records, list) or not records:
                continue
            record = records[0]
            if not isinstance(record, dict):
                continue
            latent_mu = record.get("latent_mu")
            cond_patch = record.get("cond_patch")
            cond_global = record.get("cond_global")
            inferred["z_dim"] = int(latent_mu.numel()) if torch.is_tensor(latent_mu) else 0
            inferred["cond_dim"] = int(cond_patch.shape[1]) if torch.is_tensor(cond_patch) and cond_patch.ndim == 2 else 0
            inferred["cond_global_dim"] = (
                int(cond_global.numel()) if torch.is_tensor(cond_global) and cond_global.ndim == 1 else 0
            )
            inferred["has_decoder_aux_tensors"] = bool(
                torch.is_tensor(record.get("X"))
                and torch.is_tensor(record.get("W"))
                and torch.is_tensor(record.get("x_mask"))
                and torch.is_tensor(record.get("d_in_mask"))
                and torch.is_tensor(record.get("d_out_mask"))
            )
            break
        self._record_schema_cache = dict(inferred)
        return dict(inferred)


def collate_big_vae_latent_diffusion_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("items must be non-empty")
    batch = len(items)
    first_latent = items[0].get("latent_mu")
    first_cond_patch = items[0].get("cond_patch")
    if not torch.is_tensor(first_latent) or first_latent.ndim != 1:
        raise TypeError("each item must contain rank-1 tensor 'latent_mu'")
    if not torch.is_tensor(first_cond_patch) or first_cond_patch.ndim != 2:
        raise TypeError("each item must contain rank-2 tensor 'cond_patch'")
    z_dim = int(first_latent.numel())
    cond_dim = int(first_cond_patch.shape[1])
    max_t = max(int(item["cond_patch"].shape[0]) for item in items if torch.is_tensor(item.get("cond_patch")))

    latent_mu = torch.zeros(batch, z_dim, dtype=torch.float32)
    latent_logvar = torch.zeros(batch, z_dim, dtype=torch.float32)
    cond_patch = torch.zeros(batch, max_t, cond_dim, dtype=torch.float32)
    patch_mask = torch.zeros(batch, max_t, dtype=torch.bool)
    has_cond_global = any(torch.is_tensor(item.get("cond_global")) for item in items)
    cond_global: torch.Tensor | None = None
    if has_cond_global:
        if not all(torch.is_tensor(item.get("cond_global")) for item in items):
            raise ValueError("cond_global must be present for every item in the batch or none")
        first_cond_global = items[0].get("cond_global")
        assert torch.is_tensor(first_cond_global)
        cond_global_dim = int(first_cond_global.numel())
        cond_global = torch.zeros(batch, cond_global_dim, dtype=torch.float32)
    model_names: list[str] = []
    layer_names: list[str] = []
    source_keys: list[str] = []
    metas: list[dict[str, Any]] = []
    d_in_list: list[int] = []
    d_out_list: list[int] = []
    layer_type_ids: list[int] = []
    layer_depths: list[float] = []
    has_decoder_aux_tensors = any(torch.is_tensor(item.get("W")) or torch.is_tensor(item.get("X")) for item in items)
    X: torch.Tensor | None = None
    W: torch.Tensor | None = None
    x_mask: torch.Tensor | None = None
    d_in_mask: torch.Tensor | None = None
    d_out_mask: torch.Tensor | None = None
    if has_decoder_aux_tensors:
        if not all(
            torch.is_tensor(item.get("W"))
            and torch.is_tensor(item.get("X"))
            and torch.is_tensor(item.get("x_mask"))
            and torch.is_tensor(item.get("d_in_mask"))
            and torch.is_tensor(item.get("d_out_mask"))
            for item in items
        ):
            raise ValueError("decoder auxiliary tensors must be present for every item in the batch or none")
        max_rows = max(int(item["X"].shape[0]) for item in items)
        max_d_in = max(int(item["W"].shape[0]) for item in items)
        max_d_out = max(int(item["W"].shape[1]) for item in items)
        X = torch.zeros(batch, max_rows, max_d_in, dtype=torch.float32)
        W = torch.zeros(batch, max_d_in, max_d_out, dtype=torch.float32)
        x_mask = torch.zeros(batch, max_rows, dtype=torch.bool)
        d_in_mask = torch.zeros(batch, max_d_in, dtype=torch.bool)
        d_out_mask = torch.zeros(batch, max_d_out, dtype=torch.bool)

    for idx, item in enumerate(items):
        item_latent = item.get("latent_mu")
        item_cond_patch = item.get("cond_patch")
        if not torch.is_tensor(item_latent) or not torch.is_tensor(item_cond_patch):
            raise TypeError("each item must contain tensor keys 'latent_mu' and 'cond_patch'")
        if item_latent.ndim != 1 or int(item_latent.numel()) != z_dim:
            raise ValueError(f"latent_mu shape mismatch at item {idx}: expected {(z_dim,)}, got {tuple(item_latent.shape)}")
        if item_cond_patch.ndim != 2 or int(item_cond_patch.shape[1]) != cond_dim:
            raise ValueError(
                f"cond_patch shape mismatch at item {idx}: expected (*,{cond_dim}), got {tuple(item_cond_patch.shape)}"
            )
        current_t = int(item_cond_patch.shape[0])
        latent_mu[idx] = _prepare_cpu_tensor(item_latent)
        item_logvar = item.get("latent_logvar")
        if torch.is_tensor(item_logvar):
            latent_logvar[idx] = _prepare_cpu_tensor(item_logvar)
        cond_patch[idx, :current_t] = _prepare_cpu_tensor(item_cond_patch)
        patch_mask[idx, :current_t] = True
        if has_cond_global:
            item_cond_global = item.get("cond_global")
            assert cond_global is not None
            if not torch.is_tensor(item_cond_global) or item_cond_global.ndim != 1 or int(item_cond_global.numel()) != int(cond_global.shape[1]):
                raise ValueError(
                    f"cond_global shape mismatch at item {idx}: expected {(int(cond_global.shape[1]),)}, "
                    f"got {tuple(item_cond_global.shape) if torch.is_tensor(item_cond_global) else type(item_cond_global)!r}"
                )
            cond_global[idx] = _prepare_cpu_tensor(item_cond_global)
        model_names.append(str(item.get("model_name", "")))
        layer_name = str(item.get("layer_name", ""))
        layer_names.append(layer_name)
        source_keys.append(str(item.get("source_key", "")))
        metas.append(dict(item.get("meta", {}) or {}))
        d_in_list.append(int(item.get("d_in", 0)))
        d_out_list.append(int(item.get("d_out", 0)))
        layer_type_name = str(item.get("layer_type", "")).strip() or infer_layer_type(layer_name)
        layer_depth_value = item.get("layer_depth", infer_layer_depth(layer_name))
        layer_type_ids.append(latent_diffusion_layer_type_to_id(layer_type_name))
        layer_depths.append(float(layer_depth_value) if layer_depth_value is not None else -1.0)
        if has_decoder_aux_tensors:
            item_X = _prepare_cpu_tensor(item["X"])
            item_W = _prepare_cpu_tensor(item["W"])
            item_x_mask = item["x_mask"].detach().to(device="cpu", dtype=torch.bool).contiguous()
            item_d_in_mask = item["d_in_mask"].detach().to(device="cpu", dtype=torch.bool).contiguous()
            item_d_out_mask = item["d_out_mask"].detach().to(device="cpu", dtype=torch.bool).contiguous()
            rows = int(item_X.shape[0])
            d_in = int(item_W.shape[0])
            d_out = int(item_W.shape[1])
            assert X is not None and W is not None and x_mask is not None and d_in_mask is not None and d_out_mask is not None
            X[idx, :rows, :d_in] = item_X
            W[idx, :d_in, :d_out] = item_W
            x_mask[idx, :rows] = item_x_mask
            d_in_mask[idx, :d_in] = item_d_in_mask
            d_out_mask[idx, :d_out] = item_d_out_mask

    batch_payload = {
        "latent_mu": latent_mu,
        "latent_logvar": latent_logvar,
        "cond_patch": cond_patch,
        "patch_mask": patch_mask,
        "model_names": model_names,
        "layer_names": layer_names,
        "source_keys": source_keys,
        "meta": metas,
        "d_in": torch.tensor(d_in_list, dtype=torch.long),
        "d_out": torch.tensor(d_out_list, dtype=torch.long),
        "layer_type_ids": torch.tensor(layer_type_ids, dtype=torch.long),
        "layer_depths": torch.tensor(layer_depths, dtype=torch.float32),
    }
    if cond_global is not None:
        batch_payload["cond_global"] = cond_global
    if has_decoder_aux_tensors:
        assert X is not None and W is not None and x_mask is not None and d_in_mask is not None and d_out_mask is not None
        batch_payload["X"] = X
        batch_payload["W"] = W
        batch_payload["x_mask"] = x_mask
        batch_payload["d_in_mask"] = d_in_mask
        batch_payload["d_out_mask"] = d_out_mask
    return batch_payload


def _pad_source_samples(samples: Sequence[Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    max_rows = max(int(sample.x.shape[0]) for sample in samples)
    max_d_in = max(int(sample.weight.shape[0]) for sample in samples)
    max_d_out = max(int(sample.weight.shape[1]) for sample in samples)
    batch = len(samples)

    X = torch.zeros(batch, max_rows, max_d_in, dtype=torch.float32)
    W = torch.zeros(batch, max_d_in, max_d_out, dtype=torch.float32)
    x_mask = torch.zeros(batch, max_rows, dtype=torch.bool)
    d_in_mask = torch.zeros(batch, max_d_in, dtype=torch.bool)
    d_out_mask = torch.zeros(batch, max_d_out, dtype=torch.bool)
    model_names: list[str] = []
    layer_names: list[str] = []
    metas: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        x = _prepare_cpu_tensor(sample.x)
        weight = _prepare_cpu_tensor(sample.weight)
        rows, d_in = x.shape
        _, d_out = weight.shape
        X[idx, :rows, :d_in] = x
        W[idx, :d_in, :d_out] = weight
        x_mask[idx, :rows] = True
        d_in_mask[idx, :d_in] = True
        d_out_mask[idx, :d_out] = True
        model_names.append(str(getattr(sample, "model_name", "")))
        layer_names.append(str(getattr(sample, "layer_name", "")))
        metas.append(dict(getattr(sample, "meta", {}) or {}))

    return {
        "X": X,
        "W": W,
        "x_mask": x_mask,
        "d_in_mask": d_in_mask,
        "d_out_mask": d_out_mask,
        "model_names": model_names,
        "layer_names": layer_names,
        "meta": metas,
    }


def build_big_vae_latent_diffusion_offline_dataset(
    cfg: DictConfig,
    *,
    big_vae: Any,
    dataset_iter: Iterable[Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("latent_diffusion_dataset", {})
    if not isinstance(dataset_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset must be a mapping")
    builder_cfg = dataset_cfg.get("builder", {})
    if builder_cfg is None:
        builder_cfg = {}
    if not isinstance(builder_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset.builder must be a mapping")

    root_dir = str(dataset_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("latent_diffusion_dataset.root_dir must be set")
    if not bool(getattr(big_vae, "use_distribution_encoder", False)):
        raise ValueError("latent diffusion dataset build requires frozen BigVAE with distribution encoder enabled")
    target = _resolve_explicit_slice_target(builder_cfg, patch_size=int(big_vae.cfg.patch_size))

    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    writer = BigVAELatentDiffusionOfflineWriter(
        root_dir=root_dir,
        overwrite_existing=bool(builder_cfg.get("overwrite_existing", False)),
        chunk_size_records=int(builder_cfg.get("chunk_size_records", 128)),
        patch_size=int(target.patch_size),
        target_T_patches=int(target.target_T_patches),
        target_d_out=int(target.target_d_out),
        store_decoder_aux_tensors=bool(builder_cfg.get("store_decoder_aux_tensors", False)),
        logger=logger,
        config_snapshot=cfg_snapshot if isinstance(cfg_snapshot, dict) else {},
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_latent_diffusion_offline")
    batch_size = max(1, int(builder_cfg.get("batch_size", 8)))
    encode_batch_size = max(1, int(builder_cfg.get("encode_batch_size", 1)))
    store_decoder_aux_tensors = bool(builder_cfg.get("store_decoder_aux_tensors", False))
    log_every_batches = max(1, int(builder_cfg.get("log_every_batches", 100)))
    max_records = max(0, int(builder_cfg.get("max_records", 0)))
    distribution_match_cfg = _resolve_distribution_match_cfg(builder_cfg)
    max_latent_records = int(distribution_match_cfg.max_latent_records)
    seed = int(builder_cfg.get("seed", dataset_cfg.get("source", {}).get("seed", 42)))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    sampling_rng = random.Random(seed)
    model_device = next(big_vae.parameters()).device

    pending: list[Any] = []
    total_seen = 0
    total_shape_incompatible = 0
    source_records_seen = 0
    compatible_source_records = 0
    available_latent_records = 0
    target_latent_records = 0

    def _latent_limit_reached() -> bool:
        return max_latent_records > 0 and int(writer._accepted_records) >= max_latent_records

    def _encode_and_ingest_batch_payload(batch_payload: Mapping[str, Any]) -> int:
        if _latent_limit_reached():
            return 0
        with torch.no_grad():
            encoded = encode_big_vae_layer_batch(
                big_vae,
                W=batch_payload["W"].to(device=model_device),
                X=batch_payload["X"].to(device=model_device),
                x_mask=batch_payload["x_mask"].to(device=model_device),
                d_in_mask=batch_payload["d_in_mask"].to(device=model_device),
                d_out_mask=batch_payload["d_out_mask"].to(device=model_device),
            )
        cond_patch = encoded["cond_patch"]
        dist_var_pooled = encoded.get("dist_var_pooled")
        patch_mask = encoded["patch_mask"]
        latent_mu = encoded["latent_mu"]
        latent_logvar = encoded["latent_logvar"]
        if not torch.is_tensor(cond_patch) or not torch.is_tensor(patch_mask):
            raise RuntimeError("BigVAE latent diffusion build expected tensor cond_patch and patch_mask")
        if not torch.is_tensor(latent_mu) or not torch.is_tensor(latent_logvar):
            raise RuntimeError("BigVAE latent diffusion build expected tensor latent_mu and latent_logvar")

        batch = int(latent_mu.shape[0])
        remaining_emit = batch
        if max_latent_records > 0:
            remaining_emit = min(batch, max(0, max_latent_records - int(writer._accepted_records)))
        if remaining_emit <= 0:
            return 0

        for idx in range(remaining_emit):
            valid_t = int(patch_mask[idx].to(dtype=torch.long).sum().item())
            valid_rows = int(batch_payload["x_mask"][idx].to(dtype=torch.long).sum().item())
            valid_d_in = int(batch_payload["d_in_mask"][idx].to(dtype=torch.long).sum().item())
            valid_d_out = int(batch_payload["d_out_mask"][idx].to(dtype=torch.long).sum().item())
            cond_global = None
            if torch.is_tensor(dist_var_pooled):
                cond_global_full = build_cond_global_from_dist_var_pooled(
                    dist_var_pooled=dist_var_pooled[idx : idx + 1],
                    patch_mask=patch_mask[idx : idx + 1],
                )
                if cond_global_full is not None:
                    cond_global = cond_global_full[0]
            record = {
                "source_key": _source_key(batch_payload["model_names"][idx], batch_payload["layer_names"][idx]),
                "model_name": batch_payload["model_names"][idx],
                "layer_name": batch_payload["layer_names"][idx],
                "layer_type": infer_layer_type(batch_payload["layer_names"][idx]),
                "layer_depth": infer_layer_depth(batch_payload["layer_names"][idx]),
                "d_in": valid_d_in,
                "d_out": valid_d_out,
                "cond_patch": cond_patch[idx, :valid_t],
                "patch_mask": torch.ones(valid_t, dtype=torch.bool),
                "cond_global": cond_global,
                "latent_mu": latent_mu[idx],
                "latent_logvar": latent_logvar[idx],
                "meta": batch_payload["meta"][idx],
                "target_T_patches": int(target.target_T_patches),
                "target_d_out": int(target.target_d_out),
            }
            if store_decoder_aux_tensors:
                record["X"] = batch_payload["X"][idx, :valid_rows, :valid_d_in]
                record["W"] = batch_payload["W"][idx, :valid_d_in, :valid_d_out]
                record["x_mask"] = batch_payload["x_mask"][idx, :valid_rows]
                record["d_in_mask"] = batch_payload["d_in_mask"][idx, :valid_d_in]
                record["d_out_mask"] = batch_payload["d_out_mask"][idx, :valid_d_out]
            writer.ingest(record)
        return int(remaining_emit)

    def _encode_and_ingest_samples(samples: Sequence[Any]) -> int:
        if not samples or _latent_limit_reached():
            return 0
        return _encode_and_ingest_batch_payload(_pad_source_samples(samples))

    def _flush_pending_exhaustive() -> None:
        nonlocal pending, total_shape_incompatible
        if not pending:
            return
        source_states: list[_LatentDiffusionSourceState] = []
        for raw_sample in pending:
            state = _make_latent_diffusion_source_state(raw_sample, target=target, rng=rng)
            if state is None:
                total_shape_incompatible += 1
                continue
            source_states.append(state)
        if not source_states:
            pending = []
            return
        start_offset = 0
        while source_states and not _latent_limit_reached():
            emit_batch_size = min(
                int(encode_batch_size),
                int(_latent_diffusion_source_states_total_remaining_slices(source_states)),
            )
            if max_latent_records > 0:
                emit_batch_size = min(emit_batch_size, max(0, max_latent_records - int(writer._accepted_records)))
            if emit_batch_size <= 0:
                break
            batch_payload, start_offset, _used_source_indices = _build_latent_diffusion_batch_from_source_states(
                source_states,
                batch_size=emit_batch_size,
                start_offset=start_offset,
            )
            _encode_and_ingest_batch_payload(batch_payload)
            source_states, start_offset = _prune_exhausted_latent_diffusion_source_states_with_offset(
                list(source_states),
                start_offset,
            )
        pending = []

    if distribution_match_cfg.mode == "source_proportional_joint":
        if iter(dataset_iter) is dataset_iter:
            raise TypeError(
                "source_proportional_joint requires a reiterable dataset object, not a one-shot iterator"
            )

        group_source_counts: OrderedDict[tuple[str, ...], int] = OrderedDict()
        group_capacity_totals: OrderedDict[tuple[str, ...], int] = OrderedDict()
        scan_log_every = max(1, batch_size * log_every_batches)
        for sample in dataset_iter:
            total_seen += 1
            info = _distribution_match_source_info(
                sample,
                target=target,
                group_keys=distribution_match_cfg.group_keys,
                max_slices_per_source_record=distribution_match_cfg.max_slices_per_source_record,
            )
            if info is None:
                total_shape_incompatible += 1
            else:
                group_key, effective_capacity = info
                compatible_source_records += 1
                group_source_counts[group_key] = int(group_source_counts.get(group_key, 0)) + 1
                group_capacity_totals[group_key] = int(group_capacity_totals.get(group_key, 0)) + int(effective_capacity)
            if total_seen % scan_log_every == 0:
                logger_local.info(
                    "Latent diffusion dataset prescan: seen=%s compatible=%s skipped_shape_incompatible=%s "
                    "groups=%s mode=%s root=%s",
                    int(total_seen),
                    int(compatible_source_records),
                    int(total_shape_incompatible),
                    int(len(group_source_counts)),
                    distribution_match_cfg.mode,
                    root_dir,
                )
            if max_records > 0 and total_seen >= max_records:
                break

        source_records_seen = int(total_seen)
        available_latent_records = int(sum(int(value) for value in group_capacity_totals.values()))
        target_latent_records = int(available_latent_records)
        if max_latent_records > 0:
            target_latent_records = min(int(target_latent_records), int(max_latent_records))
        if target_latent_records <= 0:
            raise RuntimeError(
                "Latent diffusion dataset prescan found no compatible latent records under the requested shape/limit: "
                f"target_T_patches={target.target_T_patches} target_d_out={target.target_d_out} "
                f"max_latent_records={max_latent_records}"
            )

        group_quota_totals = _allocate_capped_proportional_quotas(
            weights=group_source_counts,
            capacities=group_capacity_totals,
            total_target=target_latent_records,
        )
        group_quota_remaining = {group_key: int(value) for group_key, value in group_quota_totals.items()}
        group_capacity_remaining = {group_key: int(value) for group_key, value in group_capacity_totals.items()}
        selected_pending: list[SharedSample] = []
        selected_source_records = 0
        second_pass_seen = 0

        def _flush_selected_pending(*, force: bool = False) -> None:
            nonlocal selected_pending
            while selected_pending and (force or len(selected_pending) >= encode_batch_size) and not _latent_limit_reached():
                current_batch = selected_pending[:encode_batch_size]
                del selected_pending[: len(current_batch)]
                _encode_and_ingest_samples(current_batch)
            if _latent_limit_reached():
                selected_pending = []

        for sample in dataset_iter:
            second_pass_seen += 1
            info = _distribution_match_source_info(
                sample,
                target=target,
                group_keys=distribution_match_cfg.group_keys,
                max_slices_per_source_record=distribution_match_cfg.max_slices_per_source_record,
            )
            if info is not None:
                group_key, effective_capacity = info
                group_success_remaining = int(group_quota_remaining.get(group_key, 0))
                group_population_remaining = int(group_capacity_remaining.get(group_key, 0))
                take_n = _sample_slot_count_without_replacement(
                    num_slots=effective_capacity,
                    success_remaining=group_success_remaining,
                    population_remaining=group_population_remaining,
                    rng=sampling_rng,
                )
                group_quota_remaining[group_key] = max(0, group_success_remaining - int(take_n))
                group_capacity_remaining[group_key] = max(0, group_population_remaining - int(effective_capacity))
                if take_n > 0:
                    state = _make_latent_diffusion_source_state(sample, target=target, rng=rng)
                    if state is None:
                        raise RuntimeError(
                            "Latent diffusion source sample became incompatible between prescan and second pass"
                        )
                    selected_source_records += 1
                    for _ in range(int(take_n)):
                        sliced = _consume_slice_from_latent_diffusion_source_state(state)
                        if sliced is None:
                            raise RuntimeError(
                                "Latent diffusion source state exhausted before satisfying allocated slice quota"
                            )
                        selected_pending.append(sliced)
                    _flush_selected_pending()
            if second_pass_seen % scan_log_every == 0:
                logger_local.info(
                    "Latent diffusion dataset build progress: source_seen=%s accepted=%s target_latent_records=%s "
                    "selected_source_records=%s mode=%s root=%s",
                    int(second_pass_seen),
                    int(writer._accepted_records),
                    int(target_latent_records),
                    int(selected_source_records),
                    distribution_match_cfg.mode,
                    root_dir,
                )
            if _latent_limit_reached() or sum(group_quota_remaining.values()) <= 0:
                break
            if max_records > 0 and second_pass_seen >= max_records:
                break

        _flush_selected_pending(force=True)
        remaining_group_quota = int(sum(group_quota_remaining.values()))
        if remaining_group_quota > 0:
            logger_local.warning(
                "Latent diffusion proportional build finished with unsatisfied group quota: remaining=%s "
                "accepted=%s target_latent_records=%s root=%s",
                remaining_group_quota,
                int(writer._accepted_records),
                int(target_latent_records),
                root_dir,
            )
    else:
        for sample in dataset_iter:
            pending.append(sample)
            total_seen += 1
            if len(pending) >= batch_size:
                _flush_pending_exhaustive()
                processed_batches = total_seen // batch_size
                if processed_batches % log_every_batches == 0:
                    logger_local.info(
                        "Latent diffusion dataset build progress: seen=%s accepted=%s skipped_shape_incompatible=%s "
                        "target_T_patches=%s target_d_out=%s mode=%s root=%s",
                        total_seen,
                        int(writer._accepted_records),
                        int(total_shape_incompatible),
                        int(target.target_T_patches),
                        int(target.target_d_out),
                        distribution_match_cfg.mode,
                        root_dir,
                    )
            if _latent_limit_reached() or (max_records > 0 and total_seen >= max_records):
                break

        if not _latent_limit_reached():
            _flush_pending_exhaustive()
        source_records_seen = int(total_seen)

    summary = writer.close()
    summary["skipped_shape_incompatible"] = int(total_shape_incompatible)
    summary["source_records_seen"] = int(source_records_seen or total_seen)
    summary["distribution_match_mode"] = str(distribution_match_cfg.mode)
    summary["distribution_match_group_keys"] = list(distribution_match_cfg.group_keys)
    summary["max_latent_records"] = int(max_latent_records)
    summary["max_slices_per_source_record"] = int(distribution_match_cfg.max_slices_per_source_record)
    summary["compatible_source_records"] = int(compatible_source_records)
    summary["available_latent_records"] = int(available_latent_records)
    summary["target_latent_records"] = int(target_latent_records or summary.get("accepted_records", 0))
    logger_local.info(
        "Latent diffusion dataset build complete: root=%s accepted_records=%s skipped_shape_incompatible=%s "
        "source_records_seen=%s target_latent_records=%s target_T_patches=%s target_d_out=%s mode=%s actual_size_gb=%.2f",
        root_dir,
        int(summary.get("accepted_records", 0)),
        int(total_shape_incompatible),
        int(summary.get("source_records_seen", 0)),
        int(summary.get("target_latent_records", 0)),
        int(target.target_T_patches),
        int(target.target_d_out),
        distribution_match_cfg.mode,
        float(summary.get("actual_size_gb", 0.0)),
    )
    return summary


@contextlib.contextmanager
def offline_big_vae_latent_diffusion_data_pipeline(
    cfg: DictConfig,
    *,
    logger: logging.Logger | None = None,
) -> Iterator[OfflineBigVAELatentDiffusionDataset]:
    train_dataset_cfg = cfg.get("train", {}).get("dataset", {})
    if train_dataset_cfg is None:
        train_dataset_cfg = {}
    if not isinstance(train_dataset_cfg, (dict, DictConfig)):
        raise TypeError("train.dataset must be a mapping")
    root_dir = str(train_dataset_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("train.dataset.root_dir must be set")
    dataset = OfflineBigVAELatentDiffusionDataset(
        root_dir=root_dir,
        shuffle_chunks=bool(train_dataset_cfg.get("shuffle_chunks", True)),
        shuffle_records_within_chunk=bool(train_dataset_cfg.get("shuffle_records_within_chunk", True)),
        repeat=bool(train_dataset_cfg.get("repeat", True)),
        seed=int(train_dataset_cfg.get("seed", cfg.get("data", {}).get("seed", 42))),
        chunk_cache_size=int(train_dataset_cfg.get("chunk_cache_size", 4)),
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_latent_diffusion_offline")
    summary = dataset.summary()
    logger_local.info(
        "Offline latent diffusion dataset ready: root=%s accepted_records=%s z_dim=%s cond_dim=%s has_decoder_aux_tensors=%s",
        summary.get("root_dir", root_dir),
        int(summary.get("accepted_records", 0)),
        int(summary.get("z_dim", 0)),
        int(summary.get("cond_dim", 0)),
        bool(summary.get("has_decoder_aux_tensors", False)),
    )
    try:
        yield dataset
    finally:
        dataset.close()


__all__ = [
    "BigVAELatentDiffusionOfflineWriter",
    "OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION",
    "OfflineBigVAELatentDiffusionDataset",
    "build_big_vae_latent_diffusion_offline_dataset",
    "collate_big_vae_latent_diffusion_batch",
    "offline_big_vae_latent_diffusion_data_pipeline",
]
