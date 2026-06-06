from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from big_vae.datasets.offline import infer_layer_depth, infer_layer_type
from dataset.shared.types import SharedSample

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
