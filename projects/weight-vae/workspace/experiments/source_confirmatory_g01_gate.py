from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import logging
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from big_vae.datasets.offline import OfflineBigVAEDataset
from big_vae.models import build_weight_quantile_vae
from training.big_vae.checkpointing import _normalize_model_state_dict_keys
from training.big_vae.runtime import _autocast_context, _build_model_cfg, _resolve_amp


ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ROLE_SHAPES = {
    "attn_query": (768, 768),
    "attn_key": (768, 768),
    "attn_value": (768, 768),
    "attn_output": (768, 768),
    "ffn_up": (768, 3072),
    "ffn_down": (3072, 768),
}
METHODS = (
    "ae_global_c0",
    "ae_cell_mean_c0",
    "ae_cell_medoid_c0",
    "ae_native_c",
    "ae_zero_c",
    "identity",
    "zero",
)
FIXED_METHODS = METHODS[:3]
DEFAULT_CHECKPOINT = Path(
    "/home/coder/project/artifacts/training/checkpoints/"
    "weight_quantile_vae_gpu0_square/stage_1/latest.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00"
EXPECTED_CHECKPOINT_STEP = 480000
DEFAULT_SOURCE_ROOT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/training/checkpoints/"
    "weight_quantile_vae/stage_1/offline_dataset"
)
DEFAULT_HELDOUT_ROOT = Path(
    "/home/coder/project/projects/weight-vae/workspace/post_train_research/"
    "big_vae_heldout_eval/artifacts/offline_dataset"
)
DEFAULT_OUTPUT = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_confirmatory_gate_20260816"
)


@dataclass(frozen=True)
class RecordRef:
    chunk_idx: int
    record_idx: int
    source_key: str
    model_name: str
    layer_name: str
    role: str
    depth: int
    weight_shape: tuple[int, int]
    primary_dataset: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_idx": self.chunk_idx,
            "record_idx": self.record_idx,
            "source_key": self.source_key,
            "model_name": self.model_name,
            "layer_name": self.layer_name,
            "role": self.role,
            "depth": self.depth,
            "weight_shape": list(self.weight_shape),
            "primary_dataset": self.primary_dataset,
        }


@dataclass
class MatrixRecord:
    model_name: str
    depth: int
    canonical_depth: int
    role: str
    layer_name: str
    source_key: str
    context_ref: RecordRef
    score_ref: RecordRef
    W: torch.Tensor
    X_context: torch.Tensor
    X_score: torch.Tensor


@dataclass(frozen=True)
class Tiling:
    seed: int
    rows: tuple[torch.Tensor, ...]
    cols: tuple[torch.Tensor, ...]

    @property
    def num_tiles(self) -> int:
        return len(self.rows) * len(self.cols)


def stable_hex(*items: Any) -> str:
    value = "|".join(str(item) for item in items).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def stable_seed(*items: Any) -> int:
    return int(stable_hex(*items)[:16], 16) % (2**63 - 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(value.shape)).encode("utf-8"))
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ref_hash(ref: RecordRef, seed: int, panel: str) -> str:
    return stable_hex(seed, panel, ref.source_key, ref.chunk_idx, ref.record_idx)


def corrected_role(model_name: str, layer_name: str) -> str | None:
    """Resolve only explicit core paths; never trust the noisy layer_type field."""
    name = layer_name.lower()
    model = model_name.lower()
    if model == "mask2former_swin_base" and "pixel_level_module.encoder.encoder.layers." not in name:
        return None
    if name.endswith((".attention.attention.query", ".attention.self.query", ".self_attn.q_proj")):
        return "attn_query"
    if name.endswith((".attention.attention.key", ".attention.self.key", ".self_attn.k_proj")):
        return "attn_key"
    if name.endswith((".attention.attention.value", ".attention.self.value", ".self_attn.v_proj")):
        return "attn_value"
    if name.endswith((".attention.output.dense", ".self_attn.out_proj", ".self_attn.projection")):
        return "attn_output"
    if name.endswith((".intermediate.dense", ".mlp.fc1")):
        return "ffn_up"
    if name.endswith(".mlp.fc2"):
        return "ffn_down"
    if name.endswith(".output.dense") and not name.endswith(".attention.output.dense"):
        return "ffn_down"
    # Fused QKV (notably BLIP) is deliberately excluded: it cannot supply
    # separate Q/K/V cells without a parameterization-changing split.
    return None


def parse_ref(record: dict[str, Any], chunk_idx: int) -> RecordRef | None:
    model_name = str(record.get("model_name", ""))
    layer_name = str(record.get("layer_name", ""))
    role = corrected_role(model_name, layer_name)
    if role is None:
        return None
    depth_raw = record.get("layer_depth")
    if depth_raw is None:
        return None
    shape = tuple(int(v) for v in record.get("weight_shape", ()))
    if len(shape) != 2 or min(shape) < 64:
        return None
    return RecordRef(
        chunk_idx=int(chunk_idx),
        record_idx=int(record["record_idx"]),
        source_key=str(record["source_key"]),
        model_name=model_name,
        layer_name=layer_name,
        role=role,
        depth=int(depth_raw),
        weight_shape=(shape[0], shape[1]),
        primary_dataset=str(record.get("primary_dataset", "")),
    )


def scan_index(
    root: Path,
    *,
    keep_model: str | None,
    seed: int,
    logger: logging.Logger,
) -> tuple[dict[str, list[RecordRef]], dict[str, RecordRef], dict[str, Any]]:
    refs_by_source: dict[str, list[RecordRef]] = defaultdict(list)
    best_by_source: dict[str, tuple[str, RecordRef]] = {}
    paths = sorted((root / "chunk_index").glob("x_chunk_*.json"))
    if not paths:
        raise FileNotFoundError(f"no chunk indexes under {root}")
    models: set[str] = set()
    all_record_count = 0
    core_record_count = 0
    metadata_role_mismatches = 0
    corrected_role_counts: dict[str, int] = defaultdict(int)
    for path_idx, path in enumerate(paths):
        chunk_idx = int(path.stem.rsplit("_", 1)[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_chunk_id = str(payload.get("chunk_id", chunk_idx))
        payload_chunk_idx = int(payload_chunk_id.rsplit("_", 1)[-1])
        if payload_chunk_idx != chunk_idx:
            raise RuntimeError(f"chunk id/path mismatch: {path}")
        for raw in payload.get("records", []):
            all_record_count += 1
            models.add(str(raw.get("model_name", "")))
            if keep_model is not None and str(raw.get("model_name", "")) != keep_model:
                continue
            ref = parse_ref(raw, chunk_idx)
            if ref is None:
                continue
            core_record_count += 1
            corrected_role_counts[ref.role] += 1
            if str(raw.get("layer_type", "")) != ref.role:
                metadata_role_mismatches += 1
            refs_by_source[ref.source_key].append(ref)
            key = ref_hash(ref, seed, "source_best")
            previous = best_by_source.get(ref.source_key)
            if previous is None or key < previous[0]:
                best_by_source[ref.source_key] = (key, ref)
        if path_idx == 0 or (path_idx + 1) % 500 == 0 or path_idx + 1 == len(paths):
            logger.info(
                "stage=index_scan root=%s files=%s/%s all_records=%s core_records=%s",
                root,
                path_idx + 1,
                len(paths),
                all_record_count,
                core_record_count,
            )
    best = {key: value[1] for key, value in best_by_source.items()}
    audit = {
        "root": str(root.resolve()),
        "index_files": len(paths),
        "all_records": all_record_count,
        "core_records_considered": core_record_count,
        "models": sorted(models),
        "core_unique_sources": len(best),
        "corrected_role_record_counts": dict(sorted(corrected_role_counts.items())),
        "metadata_layer_type_mismatches_within_allowlisted_core": metadata_role_mismatches,
        "index_inventory_sha256": stable_hex(
            *[f"{path.name}:{path.stat().st_size}" for path in paths]
        ),
    }
    return refs_by_source, best, audit


def structural_block_key(ref: RecordRef) -> tuple[int, ...]:
    name = ref.layer_name.lower()
    hierarchical = re.search(r"(?:^|\.)layers\.(\d+)\.blocks\.(\d+)(?:\.|$)", name)
    if hierarchical:
        return (1, int(hierarchical.group(1)), int(hierarchical.group(2)))
    flat = re.search(r"(?:^|\.)encoder\.layer\.(\d+)(?:\.|$)", name)
    if flat:
        return (0, int(flat.group(1)))
    vision_flat = re.search(r"vision_model\.encoder\.layers\.(\d+)(?:\.|$)", name)
    if vision_flat:
        return (0, int(vision_flat.group(1)))
    raise ValueError(f"cannot resolve structural block path: {ref.model_name}/{ref.layer_name}")


def build_structure(refs: Iterable[RecordRef]) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    by_model_block: dict[tuple[str, tuple[int, ...]], dict[str, str]] = defaultdict(dict)
    for ref in refs:
        block = structural_block_key(ref)
        role_map = by_model_block[(ref.model_name, block)]
        previous = role_map.get(ref.role)
        if previous is not None and previous != ref.source_key:
            raise RuntimeError(
                f"duplicate corrected role in structural block: {ref.model_name}/{block}/{ref.role}"
            )
        role_map[ref.role] = ref.source_key
    mapping: dict[str, tuple[int, int]] = {}
    model_rows: list[dict[str, Any]] = []
    for model_name in sorted({key[0] for key in by_model_block}):
        blocks = sorted(key[1] for key in by_model_block if key[0] == model_name)
        role_sets = [tuple(sorted(by_model_block[(model_name, block)])) for block in blocks]
        if len(set(role_sets)) != 1:
            raise RuntimeError(f"inconsistent corrected role set across blocks for {model_name}: {role_sets}")
        layer_count = len(blocks)
        if layer_count < 2:
            raise RuntimeError(f"model has fewer than two structural blocks: {model_name}/{blocks}")
        for ordinal, block in enumerate(blocks):
            for source_key in by_model_block[(model_name, block)].values():
                mapping[source_key] = (ordinal, layer_count)
        model_rows.append({
            "model_name": model_name,
            "structural_kind": "hierarchical_stage_block" if blocks[0][0] == 1 else "flat_layer",
            "native_layers": layer_count,
            "available_roles": list(role_sets[0]),
            "block_keys": [list(block) for block in blocks],
        })
    return mapping, {"models": model_rows}


def canonical_depth(ordinal: int, layer_count: int) -> int:
    if not 0 <= ordinal < layer_count or layer_count < 2:
        raise ValueError(f"invalid ordinal/layer_count: {ordinal}/{layer_count}")
    return int(round(11.0 * ordinal / (layer_count - 1)))


def select_c0_refs(
    refs_by_source: dict[str, list[RecordRef]],
    source_best: dict[str, RecordRef],
    *,
    structure: dict[str, tuple[int, int]],
    seed: int,
    datasets_per_source: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source_key, identity in sorted(source_best.items()):
        ordinal, layer_count = structure[source_key]
        by_dataset: dict[str, list[RecordRef]] = defaultdict(list)
        for ref in refs_by_source[source_key]:
            by_dataset[ref.primary_dataset].append(ref)
        chosen_datasets = sorted(
            by_dataset,
            key=lambda dataset: stable_hex(seed, "c0_dataset", source_key, dataset),
        )[: min(datasets_per_source, len(by_dataset))]
        if not chosen_datasets:
            raise RuntimeError(f"source cell has no activation dataset: {source_key}")
        for dataset in chosen_datasets:
            ref = min(by_dataset[dataset], key=lambda value: ref_hash(value, seed, "c0_record"))
            selected.append({
                "model_name": identity.model_name,
                "role": identity.role,
                "native_ordinal": ordinal,
                "native_layers": layer_count,
                "native_u": ordinal / (layer_count - 1),
                "dataset": dataset,
                "eligible_for_cell_template": len(by_dataset) >= datasets_per_source,
                "available_distinct_datasets": len(by_dataset),
                "ref": ref,
            })
    return selected


def select_gain_refs(
    refs: Iterable[RecordRef], *, structure: dict[str, tuple[int, int]], seed: int, per_role: int
) -> list[RecordRef]:
    by_role: dict[str, list[RecordRef]] = defaultdict(list)
    for ref in refs:
        if tuple(ref.weight_shape) == ROLE_SHAPES[ref.role]:
            by_role[ref.role].append(ref)
    selected: list[RecordRef] = []
    for role in ROLES:
        candidates = by_role[role]
        if len(candidates) < per_role:
            raise RuntimeError(f"not enough exact-shape source gain refs for {role}: {len(candidates)}")
        # Greedy hash-locked balance: first minimize already selected model and
        # normalized-depth-bin counts, then use the fixed hash as tie breaker.
        model_counts: dict[str, int] = defaultdict(int)
        bin_counts: dict[int, int] = defaultdict(int)
        pool = list(candidates)
        chosen: list[RecordRef] = []
        while len(chosen) < per_role:
            ref = min(
                pool,
                key=lambda value: (
                    model_counts[value.model_name],
                    bin_counts[canonical_depth(*structure[value.source_key])],
                    ref_hash(value, seed, "gain_panel"),
                ),
            )
            pool.remove(ref)
            chosen.append(ref)
            model_counts[ref.model_name] += 1
            bin_counts[canonical_depth(*structure[ref.source_key])] += 1
        selected.extend(chosen)
    return selected


def build_dataset(root: Path, seed: int) -> OfflineBigVAEDataset:
    return OfflineBigVAEDataset(
        root_dir=root,
        shuffle_chunks=False,
        shuffle_records_within_chunk=False,
        repeat=False,
        seed=seed,
        weight_cache_size=96,
        sampling_mode="balanced",
        sampling_group_keys=("dataset", "model"),
        sampling_window_size=2048,
        sampling_max_records_per_chunk_round=8,
        x_chunk_cache_size=3,
    )


def load_sample(dataset: OfflineBigVAEDataset, ref: RecordRef) -> tuple[torch.Tensor, torch.Tensor]:
    sample = dataset._shared_sample_from_record_ref(chunk_idx=ref.chunk_idx, record_idx=ref.record_idx)
    W = sample.weight.detach().cpu().to(torch.float32).contiguous()
    X = sample.x.detach().cpu().to(torch.float32).contiguous()
    if tuple(W.shape) != ref.weight_shape or X.ndim != 2 or int(X.shape[1]) != int(W.shape[0]):
        raise RuntimeError(f"runtime/index mismatch for {ref}: W={tuple(W.shape)} X={tuple(X.shape)}")
    return W, X


def make_partition(size: int, group_size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    if size % group_size:
        raise ValueError(f"confirmatory full coverage requires divisibility: {size} % {group_size}")
    order = torch.randperm(size, generator=generator)
    return tuple(
        order[start : start + group_size].sort().values
        for start in range(0, size, group_size)
    )


def make_tiling(record: MatrixRecord, seed: int) -> Tiling:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, record.source_key, record.role, record.depth))
    patch_groups = make_partition(int(record.W.shape[0]) // 16, 4, generator)
    offsets = torch.arange(16)
    rows = tuple((group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values for group in patch_groups)
    cols = make_partition(int(record.W.shape[1]), 64, generator)
    return Tiling(seed=seed, rows=rows, cols=cols)


def split_tiles(W: torch.Tensor, tiling: Tiling) -> torch.Tensor:
    return torch.stack([W[row][:, col] for row in tiling.rows for col in tiling.cols])


def reassemble(tiles: torch.Tensor, tiling: Tiling, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty(shape, dtype=tiles.dtype)
    coverage = torch.zeros(shape, dtype=torch.int16)
    index = 0
    for row in tiling.rows:
        for col in tiling.cols:
            result[row[:, None], col[None, :]] = tiles[index].cpu()
            coverage[row[:, None], col[None, :]] += 1
            index += 1
    if index != int(tiles.shape[0]):
        raise RuntimeError(f"tile count mismatch: consumed={index} tensor={tiles.shape[0]}")
    return result, coverage


def validate_tiling(record: MatrixRecord, tiling: Tiling) -> dict[str, Any]:
    W_tiles = split_tiles(record.W, tiling)
    W_reassembled, coverage = reassemble(W_tiles, tiling, tuple(record.W.shape))
    coordinate = torch.arange(record.W.numel(), dtype=torch.int64).reshape(record.W.shape)
    coded_reassembled, coded_coverage = reassemble(split_tiles(coordinate, tiling), tiling, tuple(record.W.shape))
    if not torch.equal(W_reassembled, record.W):
        raise RuntimeError(f"W split/scatter is not bit exact: {record.source_key}")
    if not torch.equal(coded_reassembled, coordinate):
        raise RuntimeError(f"coordinate split/scatter is not bit exact: {record.source_key}")
    if not torch.all(coverage == 1) or not torch.all(coded_coverage == 1):
        raise RuntimeError(f"coverage is not exactly one: {record.source_key}")
    partition_hash = stable_hex(
        seed_repr(tiling.seed),
        *[",".join(str(int(v)) for v in row.tolist()) for row in tiling.rows],
        *[",".join(str(int(v)) for v in col.tolist()) for col in tiling.cols],
    )
    return {
        "tiling_seed": tiling.seed,
        "depth": record.depth,
        "role": record.role,
        "source_key": record.source_key,
        "shape": "x".join(str(int(v)) for v in record.W.shape),
        "row_groups": len(tiling.rows),
        "col_groups": len(tiling.cols),
        "num_tiles": tiling.num_tiles,
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "w_reassembly_bit_exact": True,
        "coordinate_reassembly_bit_exact": True,
        "partition_sha256": partition_hash,
    }


def seed_repr(seed: int) -> str:
    return f"seed={int(seed)}"


@torch.inference_mode()
def extract_condition_vector(
    *,
    model: torch.nn.Module,
    X: torch.Tensor,
    source_key: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    d_in = int(X.shape[1])
    if d_in % 64:
        raise ValueError(f"C0 source d_in must be divisible by 64, got {d_in}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed("c0_rows", source_key))
    patch_groups = make_partition(d_in // 16, 4, generator)
    offsets = torch.arange(16)
    row_groups = [(group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values for group in patch_groups]
    var_vectors: list[torch.Tensor] = []
    patch_vectors: list[torch.Tensor] = []
    for begin in range(0, len(row_groups), 32):
        groups = row_groups[begin : begin + 32]
        X_batch = torch.stack([X[:, rows] for rows in groups]).to(device)
        x_mask = torch.ones((len(groups), int(X.shape[0])), dtype=torch.bool, device=device)
        d_in_mask = torch.ones((len(groups), 64), dtype=torch.bool, device=device)
        with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            outputs = model._encode_distribution_context(X_batch, x_mask=x_mask, d_in_mask=d_in_mask)
        _T, _d_in_pad, _patch_mask, _structural, c_var, c_patch, _pooled = outputs
        if c_var is None or c_patch is None:
            raise RuntimeError("distribution encoder returned None")
        var_vectors.append(c_var.float().mean(dim=(1, 2)).cpu())
        patch_vectors.append(c_patch.float().mean(dim=1).cpu())
    return (
        torch.cat(var_vectors).mean(dim=0),
        torch.cat(patch_vectors).mean(dim=0),
        len(row_groups),
    )


def interpolate_features(
    native_u: torch.Tensor,
    native_values: torch.Tensor,
    grid_u: torch.Tensor,
) -> torch.Tensor:
    if native_u.ndim != 1 or native_values.ndim != 2 or int(native_values.shape[0]) != int(native_u.numel()):
        raise ValueError(f"bad interpolation shapes: {native_u.shape}/{native_values.shape}")
    result: list[torch.Tensor] = []
    for target in grid_u:
        right = int(torch.searchsorted(native_u, target, right=False))
        if right <= 0:
            result.append(native_values[0])
        elif right >= int(native_u.numel()):
            result.append(native_values[-1])
        else:
            left = right - 1
            span = float(native_u[right] - native_u[left])
            alpha = 0.0 if span <= 0 else float((target - native_u[left]) / span)
            result.append((1.0 - alpha) * native_values[left] + alpha * native_values[right])
    return torch.stack(result)


def build_templates(
    vectors: list[dict[str, Any]], *, minimum_families: int = 8
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    # First average the two hash-locked datasets within each exact W source.
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in vectors:
        by_source[item["source_key"]].append(item)
    source_units: list[dict[str, Any]] = []
    for source_key, values in sorted(by_source.items()):
        datasets = sorted({value["dataset"] for value in values})
        first = values[0]
        source_units.append({
            "source_key": source_key,
            "model_name": first["model_name"],
            "role": first["role"],
            "native_ordinal": first["native_ordinal"],
            "native_layers": first["native_layers"],
            "native_u": first["native_u"],
            "datasets": datasets,
            "eligible_for_cell_template": bool(first["eligible_for_cell_template"] and len(datasets) >= 2),
            "c_var": torch.stack([value["c_var"] for value in values]).mean(0),
            "c_patch": torch.stack([value["c_patch"] for value in values]).mean(0),
        })

    grid_u = torch.linspace(0.0, 1.0, 12)
    by_model_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in source_units:
        by_model_role[(unit["model_name"], unit["role"])].append(unit)
    interpolated: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    interpolated_global: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    interpolation_rows: list[dict[str, Any]] = []
    excluded_family_role_rows: list[dict[str, Any]] = []
    for (model_name, role), values in sorted(by_model_role.items()):
        values = sorted(values, key=lambda item: int(item["native_ordinal"]))
        layer_count = int(values[0]["native_layers"])
        ordinals = [int(item["native_ordinal"]) for item in values]
        if ordinals != list(range(layer_count)):
            raise RuntimeError(f"incomplete native depth grid: {model_name}/{role}/{ordinals}/{layer_count}")
        native_u = torch.tensor([float(item["native_u"]) for item in values])
        var_interp = interpolate_features(native_u, torch.stack([item["c_var"] for item in values]), grid_u)
        patch_interp = interpolate_features(native_u, torch.stack([item["c_patch"] for item in values]), grid_u)
        for j in range(12):
            interpolated_global[(model_name, role, j)] = {"c_var": var_interp[j], "c_patch": patch_interp[j]}
        cell_eligible = all(bool(item["eligible_for_cell_template"]) for item in values)
        if cell_eligible:
            for j in range(12):
                interpolated[(model_name, role, j)] = {"c_var": var_interp[j], "c_patch": patch_interp[j]}
        else:
            excluded_family_role_rows.append({
                "model_name": model_name,
                "role": role,
                "reason": "at_least_one_native_source_unit_has_fewer_than_2_distinct_datasets",
                "undercovered_native_ordinals": [
                    int(item["native_ordinal"]) for item in values if not item["eligible_for_cell_template"]
                ],
            })
        interpolation_rows.append({
            "model_name": model_name,
            "role": role,
            "native_layers": layer_count,
            "native_source_units": len(values),
            "canonical_cells": 12,
            "native_u": [float(value) for value in native_u],
            "eligible_for_cell_template": cell_eligible,
        })

    mean_cells: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    family_count_rows: list[dict[str, Any]] = []
    for role in ROLES:
        families = sorted({model for model, candidate_role, _j in interpolated if candidate_role == role})
        if len(families) < minimum_families:
            raise RuntimeError(f"too few source families for {role}: {families}")
        for j in range(12):
            mean_cells[(role, j)] = {
                "c_var": torch.stack([interpolated[(model, role, j)]["c_var"] for model in families]).mean(0),
                "c_patch": torch.stack([interpolated[(model, role, j)]["c_patch"] for model in families]).mean(0),
            }
            family_count_rows.append({
                "role": role,
                "canonical_depth": j,
                "families": len(families),
                "family_names": families,
            })

    # Stronger global stress arm: equal role and canonical-depth weighting
    # within each model, then equal model-family weighting.
    model_globals: list[dict[str, torch.Tensor]] = []
    for model_name in sorted({key[0] for key in interpolated_global}):
        values = [value for (model, _role, _j), value in interpolated_global.items() if model == model_name]
        model_globals.append({
            "c_var": torch.stack([value["c_var"] for value in values]).mean(0),
            "c_patch": torch.stack([value["c_patch"] for value in values]).mean(0),
        })
    global_template = {
        "c_var": torch.stack([value["c_var"] for value in model_globals]).mean(0),
        "c_patch": torch.stack([value["c_patch"] for value in model_globals]).mean(0),
    }

    # Support sensitivity: one actual source unit per family nearest in native
    # normalized depth, standardized feature distance to the arithmetic cell mean.
    medoid_cells: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    medoid_manifest: list[dict[str, Any]] = []
    for role in ROLES:
        eligible_families = {model for model, candidate_role, _j in interpolated if candidate_role == role}
        role_units = [
            unit for unit in source_units
            if unit["role"] == role and unit["model_name"] in eligible_families
        ]
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in role_units:
            by_model[unit["model_name"]].append(unit)
        for j in range(12):
            target_u = j / 11.0
            candidates = [
                min(values, key=lambda unit: (abs(float(unit["native_u"]) - target_u), unit["source_key"]))
                for _model, values in sorted(by_model.items())
            ]
            concatenated = torch.stack([torch.cat([unit["c_var"], unit["c_patch"]]) for unit in candidates])
            scale = concatenated.std(dim=0, unbiased=False).clamp_min(1e-6)
            target = torch.cat([mean_cells[(role, j)]["c_var"], mean_cells[(role, j)]["c_patch"]])
            ranked = []
            for unit, vector in zip(candidates, concatenated, strict=True):
                distance = float(((vector - target) / scale).square().mean())
                ranked.append((distance, unit["source_key"], unit))
            distance, _source_key, medoid = min(ranked)
            medoid_cells[(role, j)] = {"c_var": medoid["c_var"], "c_patch": medoid["c_patch"]}
            medoid_manifest.append({
                "role": role,
                "canonical_depth": j,
                "source_key": medoid["source_key"],
                "model_name": medoid["model_name"],
                "native_ordinal": medoid["native_ordinal"],
                "native_layers": medoid["native_layers"],
                "native_u": medoid["native_u"],
                "standardized_mean_squared_distance": distance,
                "candidate_families": len(candidates),
            })
    expected = {(role, j) for role in ROLES for j in range(12)}
    if set(mean_cells) != expected or set(medoid_cells) != expected:
        raise RuntimeError(f"canonical template grid incomplete: {sorted(expected - set(mean_cells))}")
    return {
        "global": global_template,
        "cell_mean": mean_cells,
        "cell_medoid": medoid_cells,
    }, medoid_manifest, {
        "family_counts": family_count_rows,
        "interpolation": interpolation_rows,
        "excluded_family_roles": excluded_family_role_rows,
    }


def resolve_template(templates: dict[str, Any], method: str, role: str, bin_idx: int) -> dict[str, torch.Tensor]:
    if method == "ae_global_c0":
        return templates["global"]
    if method == "ae_cell_mean_c0":
        return templates["cell_mean"][(role, bin_idx)]
    if method == "ae_cell_medoid_c0":
        return templates["cell_medoid"][(role, bin_idx)]
    raise ValueError(method)


@torch.inference_mode()
def fixed_forward(model: torch.nn.Module, W: torch.Tensor, template: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = int(W.shape[0])
    d_in_mask = torch.ones((batch, 64), dtype=torch.bool, device=W.device)
    d_out_mask = torch.ones((batch, 64), dtype=torch.bool, device=W.device)
    _idx, patch_mask, _structural, _valid, T, d_in_pad = model._build_batched_patch_indices(
        d_in_mask=d_in_mask,
        patch_size=int(model.cfg.patch_size),
    )
    patch_size = int(model.cfg.patch_size)
    c_var = template["c_var"].to(device=W.device, dtype=W.dtype).view(1, 1, 1, -1).expand(
        batch, T, patch_size, -1
    )
    c_patch = template["c_patch"].to(device=W.device, dtype=W.dtype).view(1, 1, -1).expand(batch, T, -1)
    c_pooled = c_var.mean(dim=2)
    latents, encode_debug = model._encode_latent_slots(
        W,
        T=T,
        d_in_pad=d_in_pad,
        patch_mask=patch_mask,
        d_out_mask=d_out_mask,
        dist_var_by_patch=c_var,
        dist_patch_by_patch=c_patch,
        dist_var_pooled=c_pooled,
        return_debug_info=True,
    )
    W_hat, _mu, _logvar, _dirs = model._decode_from_latent_slots(
        latents,
        dist_patch_by_patch=c_patch,
        patch_mask=patch_mask,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        d_in=64,
        d_out=64,
        d_in_pad=d_in_pad,
        T=T,
        encoder_patch_tokens=encode_debug["encoder_patch_tokens"],
    )
    return W_hat


@torch.inference_mode()
def fixed_forward_with_dummy_x(
    model: torch.nn.Module,
    W: torch.Tensor,
    template: dict[str, torch.Tensor],
    dummy_x: torch.Tensor,
) -> torch.Tensor:
    # Explicit stub boundary: only the shape is validated; no dummy-X value is
    # allowed to reach the distribution encoder or weight AE.
    if dummy_x.ndim != 3 or int(dummy_x.shape[0]) != int(W.shape[0]):
        raise ValueError(f"invalid dummy-X shape: {dummy_x.shape} for W={W.shape}")
    return fixed_forward(model, W, template)


@torch.inference_mode()
def manual_native_forward(
    model: torch.nn.Module,
    W: torch.Tensor,
    X: torch.Tensor,
    *,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    """Manual internal path used only to assert parity with model.forward."""
    batch, d_in, d_out = W.shape
    T, d_in_pad, patch_mask, _structural, c_var, c_patch, c_pooled = model._encode_distribution_context(
        X,
        x_mask=x_mask,
        d_in_mask=d_in_mask,
    )
    latents, debug = model._encode_latent_slots(
        W,
        T=T,
        d_in_pad=d_in_pad,
        patch_mask=patch_mask,
        d_out_mask=model._validate_d_out_mask(d_out_mask, batch_size=batch, d_out=d_out, device=W.device),
        dist_var_by_patch=c_var,
        dist_patch_by_patch=c_patch,
        dist_var_pooled=c_pooled,
        return_debug_info=True,
    )
    W_hat, _mu, _logvar, _dirs = model._decode_from_latent_slots(
        latents,
        dist_patch_by_patch=c_patch,
        patch_mask=patch_mask,
        d_in_mask=model._validate_d_in_mask(d_in_mask, batch_size=batch, d_in=d_in, device=W.device),
        d_out_mask=model._validate_d_out_mask(d_out_mask, batch_size=batch, d_out=d_out, device=W.device),
        d_in=d_in,
        d_out=d_out,
        d_in_pad=d_in_pad,
        T=T,
        encoder_patch_tokens=debug["encoder_patch_tokens"],
    )
    return W_hat


@torch.inference_mode()
def decode_matrix(
    *,
    model: torch.nn.Module,
    record: MatrixRecord,
    tiling: Tiling,
    method: str,
    templates: dict[str, Any],
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> torch.Tensor:
    tiles = split_tiles(record.W, tiling)
    predictions: list[torch.Tensor] = []
    started = time.monotonic()
    for batch_idx, begin in enumerate(range(0, int(tiles.shape[0]), batch_size)):
        end = min(begin + batch_size, int(tiles.shape[0]))
        W_batch = tiles[begin:end].to(device)
        with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            if method in FIXED_METHODS or method == "ae_zero_c":
                template = (
                    resolve_template(templates, method, record.role, record.canonical_depth)
                    if method in FIXED_METHODS
                    else {
                        "c_var": torch.zeros_like(templates["global"]["c_var"]),
                        "c_patch": torch.zeros_like(templates["global"]["c_patch"]),
                    }
                )
                W_hat = fixed_forward(
                    model,
                    W_batch,
                    template,
                )
            elif method == "ae_native_c":
                coordinates = [
                    (row, col)
                    for row in tiling.rows
                    for col in tiling.cols
                ][begin:end]
                X_batch = torch.stack([record.X_context[:, row] for row, _col in coordinates]).to(device)
                n = int(X_batch.shape[1])
                x_mask = torch.ones((len(coordinates), n), dtype=torch.bool, device=device)
                d_in_mask = torch.ones((len(coordinates), 64), dtype=torch.bool, device=device)
                d_out_mask = torch.ones((len(coordinates), 64), dtype=torch.bool, device=device)
                W_hat, _mu, _logvar, _dirs = model(
                    W_batch,
                    X_batch,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                )
            else:
                raise ValueError(method)
        predictions.append(W_hat.float().cpu())
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or end == int(tiles.shape[0]):
            elapsed = time.monotonic() - started
            logger.info(
                "stage=ae_decode method=%s seed=%s depth=%s role=%s tiles=%s/%s rate=%.1f/s elapsed=%.1fs",
                method,
                tiling.seed,
                record.depth,
                record.role,
                end,
                int(tiles.shape[0]),
                end / max(elapsed, 1e-6),
                elapsed,
            )
    result, coverage = reassemble(torch.cat(predictions), tiling, tuple(record.W.shape))
    if not torch.all(coverage == 1) or not torch.isfinite(result).all():
        raise RuntimeError(f"invalid decoded reassembly: {method} {record.source_key}")
    return result


def sufficient_stats(W: torch.Tensor, prediction: torch.Tensor, X: torch.Tensor) -> dict[str, float]:
    W64 = W.to(torch.float64)
    P64 = prediction.to(torch.float64)
    X64 = X.to(torch.float64)
    target = X64 @ W64
    pred = X64 @ P64
    return {
        "w_target": float(W64.square().sum()),
        "w_pred": float(P64.square().sum()),
        "w_dot": float((W64 * P64).sum()),
        "w_error": float((W64 - P64).square().sum()),
        "x_target": float(target.square().sum()),
        "x_pred": float(pred.square().sum()),
        "x_dot": float((target * pred).sum()),
        "x_error": float((target - pred).square().sum()),
    }


def scaled_stats(raw: dict[str, float], gain: float) -> dict[str, float]:
    return {
        **raw,
        "w_error_scaled": raw["w_target"] - 2 * gain * raw["w_dot"] + gain * gain * raw["w_pred"],
        "x_error_scaled": raw["x_target"] - 2 * gain * raw["x_dot"] + gain * gain * raw["x_pred"],
        "w_pred_scaled": gain * gain * raw["w_pred"],
        "x_pred_scaled": gain * gain * raw["x_pred"],
        "w_dot_scaled": gain * raw["w_dot"],
        "x_dot_scaled": gain * raw["x_dot"],
    }


def fit_gains(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for method in FIXED_METHODS:
        for role in ROLES:
            group = [row for row in rows if row["method"] == method and row["role"] == role]
            dot = sum(row["x_dot"] for row in group)
            energy = sum(row["x_pred"] for row in group)
            if energy <= 0:
                raise RuntimeError(f"nonpositive gain denominator: {method}/{role}")
            result[(method, role)] = dot / energy
    return result


def metric_row(
    record: MatrixRecord,
    *,
    tiling_seed: int,
    method: str,
    prediction: torch.Tensor,
    gain: float,
    num_tiles: int,
) -> dict[str, Any]:
    raw = sufficient_stats(record.W, prediction, record.X_score)
    values = scaled_stats(raw, gain)
    eps = 1e-300
    return {
        "tiling_seed": tiling_seed,
        "method": method,
        "depth": record.depth,
        "canonical_depth": record.canonical_depth,
        "role": record.role,
        "layer_name": record.layer_name,
        "source_key": record.source_key,
        "shape": "x".join(str(int(v)) for v in record.W.shape),
        "num_tiles": num_tiles,
        "gain": gain,
        **values,
        "raw_E_W": values["w_error"] / max(values["w_target"], eps),
        "raw_E_X": values["x_error"] / max(values["x_target"], eps),
        "calibrated_E_W": values["w_error_scaled"] / max(values["w_target"], eps),
        "calibrated_E_X": values["x_error_scaled"] / max(values["x_target"], eps),
        "raw_weight_cosine": values["w_dot"] / math.sqrt(max(values["w_target"] * values["w_pred"], eps)),
        "raw_operator_cosine": values["x_dot"] / math.sqrt(max(values["x_target"] * values["x_pred"], eps)),
        "raw_weight_norm_ratio": math.sqrt(values["w_pred"] / max(values["w_target"], eps)),
        "raw_operator_norm_ratio": math.sqrt(values["x_pred"] / max(values["x_target"], eps)),
        "prediction_sha256": tensor_sha256(prediction),
    }


def ratio_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["tiling_seed"]) for row in matrix_rows}):
        for method in METHODS:
            group = [row for row in matrix_rows if int(row["tiling_seed"]) == seed and row["method"] == method]
            if len(group) != 72:
                raise RuntimeError(f"expected 72 rows: seed={seed} method={method} got={len(group)}")
            role_ratios: dict[str, dict[str, float]] = {}
            for role in ROLES:
                role_group = [row for row in group if row["role"] == role]
                sums = {key: sum(float(row[key]) for row in role_group) for key in (
                    "w_target", "w_pred", "w_dot", "w_error", "x_target", "x_pred", "x_dot", "x_error",
                    "w_error_scaled", "x_error_scaled", "w_pred_scaled", "x_pred_scaled", "w_dot_scaled", "x_dot_scaled",
                )}
                ratios = {
                    "raw_E_W": sums["w_error"] / sums["w_target"],
                    "raw_E_X": sums["x_error"] / sums["x_target"],
                    "calibrated_E_W": sums["w_error_scaled"] / sums["w_target"],
                    "calibrated_E_X": sums["x_error_scaled"] / sums["x_target"],
                    "raw_weight_cosine": sums["w_dot"] / math.sqrt(max(sums["w_target"] * sums["w_pred"], 1e-300)),
                    "raw_operator_cosine": sums["x_dot"] / math.sqrt(max(sums["x_target"] * sums["x_pred"], 1e-300)),
                    "raw_weight_norm_ratio": math.sqrt(sums["w_pred"] / sums["w_target"]),
                    "raw_operator_norm_ratio": math.sqrt(sums["x_pred"] / sums["x_target"]),
                }
                role_ratios[role] = ratios
                output.append({"tiling_seed": seed, "method": method, "aggregation": role, **ratios})
            macro = {
                key: sum(role_ratios[role][key] for role in ROLES) / len(ROLES)
                for key in next(iter(role_ratios.values()))
            }
            output.append({"tiling_seed": seed, "method": method, "aggregation": "macro", **macro})
            sums = {key: sum(float(row[key]) for row in group) for key in (
                "w_target", "w_pred", "w_dot", "w_error", "x_target", "x_pred", "x_dot", "x_error",
                "w_error_scaled", "x_error_scaled",
            )}
            micro = {
                "raw_E_W": sums["w_error"] / sums["w_target"],
                "raw_E_X": sums["x_error"] / sums["x_target"],
                "calibrated_E_W": sums["w_error_scaled"] / sums["w_target"],
                "calibrated_E_X": sums["x_error_scaled"] / sums["x_target"],
                "raw_weight_cosine": sums["w_dot"] / math.sqrt(max(sums["w_target"] * sums["w_pred"], 1e-300)),
                "raw_operator_cosine": sums["x_dot"] / math.sqrt(max(sums["x_target"] * sums["x_pred"], 1e-300)),
                "raw_weight_norm_ratio": math.sqrt(sums["w_pred"] / sums["w_target"]),
                "raw_operator_norm_ratio": math.sqrt(sums["x_pred"] / sums["x_target"]),
            }
            output.append({"tiling_seed": seed, "method": method, "aggregation": "micro", **micro})
    return output


def bootstrap_rows(matrix_rows: list[dict[str, Any]], *, draws: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, 12, size=(draws, 12))
    for tiling_seed in sorted({int(row["tiling_seed"]) for row in matrix_rows}):
        for method in METHODS:
            group = [row for row in matrix_rows if int(row["tiling_seed"]) == tiling_seed and row["method"] == method]
            by_depth_role = {(int(row["depth"]), str(row["role"])): row for row in group}
            if len(by_depth_role) != 72:
                raise RuntimeError(f"bootstrap grid incomplete: {tiling_seed}/{method}")
            for calibrated in (False, True):
                w_key = "w_error_scaled" if calibrated else "w_error"
                x_key = "x_error_scaled" if calibrated else "x_error"
                role_w = []
                role_x = []
                role_w_den = []
                role_x_den = []
                for role in ROLES:
                    role_w.append(np.array([by_depth_role[(d, role)][w_key] for d in range(12)]))
                    role_x.append(np.array([by_depth_role[(d, role)][x_key] for d in range(12)]))
                    role_w_den.append(np.array([by_depth_role[(d, role)]["w_target"] for d in range(12)]))
                    role_x_den.append(np.array([by_depth_role[(d, role)]["x_target"] for d in range(12)]))
                w_num = np.stack([arr[draw_indices].sum(axis=1) for arr in role_w], axis=1)
                x_num = np.stack([arr[draw_indices].sum(axis=1) for arr in role_x], axis=1)
                w_den = np.stack([arr[draw_indices].sum(axis=1) for arr in role_w_den], axis=1)
                x_den = np.stack([arr[draw_indices].sum(axis=1) for arr in role_x_den], axis=1)
                values = {
                    "macro_E_W": (w_num / w_den).mean(axis=1),
                    "macro_E_X": (x_num / x_den).mean(axis=1),
                    "micro_E_W": w_num.sum(axis=1) / w_den.sum(axis=1),
                    "micro_E_X": x_num.sum(axis=1) / x_den.sum(axis=1),
                }
                for metric, samples in values.items():
                    output.append({
                        "tiling_seed": tiling_seed,
                        "method": method,
                        "calibration": "calibrated" if calibrated else "raw",
                        "metric": metric,
                        "draws": draws,
                        "mean": float(samples.mean()),
                        "l95": float(np.quantile(samples, 0.05)),
                        "u95": float(np.quantile(samples, 0.95)),
                    })
    return output


def plot_results(aggregate: list[dict[str, Any]], bootstrap: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = ["ae_cell_mean_c0", "ae_global_c0", "ae_cell_medoid_c0", "ae_native_c", "ae_zero_c", "zero"]
    seeds = sorted({int(row["tiling_seed"]) for row in aggregate})
    fig, axes = plt.subplots(1, len(seeds), figsize=(16, 6), sharey=True)
    for ax, seed in zip(np.atleast_1d(axes), seeds, strict=True):
        values = []
        errors = []
        for method in methods:
            row = next(r for r in aggregate if r["tiling_seed"] == seed and r["method"] == method and r["aggregation"] == "macro")
            values.append(float(row["raw_E_X"]))
            boot = next(r for r in bootstrap if r["tiling_seed"] == seed and r["method"] == method and r["calibration"] == "raw" and r["metric"] == "macro_E_X")
            errors.append(max(0.0, float(boot["u95"]) - values[-1]))
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#aaaaaa"]
        bars = ax.bar(methods, values, yerr=errors, capsize=3, color=colors)
        ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="G1 U95 threshold")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="zero")
        ax.set_title(f"tiling seed {seed}")
        ax.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Raw full-matrix operator error, macro (one-sided U95)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Held-out ViT-B source confirmatory G0/G1")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Confirmatory held-out ViT-B source-only G0/G1 gate")
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    value.add_argument("--heldout-root", type=Path, default=DEFAULT_HELDOUT_ROOT)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--seed", type=int, default=260816)
    value.add_argument("--tiling-seeds", default="26081601,26081602")
    value.add_argument("--c0-datasets-per-source", type=int, default=2)
    value.add_argument("--gain-matrices-per-role", type=int, default=12)
    value.add_argument("--bootstrap-draws", type=int, default=10000)
    value.add_argument("--no-amp", action="store_true")
    value.add_argument("--audit-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", mode="w")],
        force=True,
    )
    logger = logging.getLogger("source_confirmatory_g01")
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tiling_seeds = tuple(int(v) for v in args.tiling_seeds.split(","))
    if len(tiling_seeds) != 2 or tiling_seeds[0] == tiling_seeds[1]:
        raise ValueError(f"exactly two distinct tiling seeds required: {tiling_seeds}")

    resolved = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_root": str(args.source_root.resolve()),
        "heldout_root": str(args.heldout_root.resolve()),
        "output_dir": str(output),
        "device": str(device),
        "dtype": "checkpoint AMP policy",
        "seed": args.seed,
        "tiling_seeds": list(tiling_seeds),
        "batch_size": args.batch_size,
        "c0_distinct_datasets_per_exact_source_cell": args.c0_datasets_per_source,
        "gain_matrices_per_role": args.gain_matrices_per_role,
        "bootstrap_draws": args.bootstrap_draws,
        "cache_mode": "offline exact-ref chunk loading; no target/data2vec access",
        "conditioning_primary": "source-only role x canonical normalized-depth(12) arithmetic cell mean",
        "conditioning_diagnostics": ["global C0 stronger stress", "role_x_depth medoid sensitivity", "Native-C"],
        "c_pooled": "derived only from broadcast c_var",
        "progress": "verbose",
        "audit_only": bool(args.audit_only),
    }
    write_json(output / "resolved_config.json", resolved)
    logger.info("resolved_config=%s", json.dumps(resolved, sort_keys=True))

    logger.info("stage=index_audit")
    source_refs_by_source, source_best, source_audit = scan_index(
        args.source_root, keep_model=None, seed=args.seed, logger=logger
    )
    heldout_refs_by_source, heldout_best, heldout_audit = scan_index(
        args.heldout_root, keep_model="vit_base_p16_224", seed=args.seed, logger=logger
    )
    source_models = sorted({ref.model_name for ref in source_best.values()})
    if len(source_models) != 11 or "vit_base_p16_224" in source_models:
        raise RuntimeError(f"source split invalid: models={source_models}")
    heldout_core = [ref for ref in heldout_best.values() if tuple(ref.weight_shape) == ROLE_SHAPES[ref.role]]
    if len(heldout_core) != 72:
        raise RuntimeError(f"expected 72 exact core heldout matrices, got {len(heldout_core)}")
    structure, structure_audit = build_structure(source_best.values())
    if set(structure) != set(source_best):
        raise RuntimeError(
            f"structural parser lost source keys: missing={len(set(source_best) - set(structure))}"
        )
    c0_selection = select_c0_refs(
        source_refs_by_source,
        source_best,
        structure=structure,
        seed=args.seed,
        datasets_per_source=args.c0_datasets_per_source,
    )
    gain_refs = select_gain_refs(
        source_best.values(), structure=structure, seed=args.seed, per_role=args.gain_matrices_per_role
    )
    eligibility_by_source = {
        item["ref"].source_key: bool(item["eligible_for_cell_template"])
        for item in c0_selection
    }
    family_role_sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source_key, ref in source_best.items():
        family_role_sources[(ref.model_name, ref.role)].append(source_key)
    eligible_family_roles: dict[str, list[str]] = defaultdict(list)
    excluded_family_roles: list[dict[str, Any]] = []
    for (model_name, role), source_keys in sorted(family_role_sources.items()):
        undercovered = sorted(key for key in source_keys if not eligibility_by_source.get(key, False))
        if undercovered:
            excluded_family_roles.append({
                "model_name": model_name,
                "role": role,
                "undercovered_source_keys": undercovered,
                "rule": "exclude entire model-family x role interpolation curve from cell mean/medoid",
            })
        else:
            eligible_family_roles[role].append(model_name)
    eligibility_audit = {
        "minimum_distinct_datasets_per_source_unit": args.c0_datasets_per_source,
        "eligible_family_roles": {role: sorted(values) for role, values in eligible_family_roles.items()},
        "eligible_family_count_per_role": {role: len(eligible_family_roles[role]) for role in ROLES},
        "excluded_family_roles": excluded_family_roles,
        "minimum_families_required": 8,
    }
    if any(len(eligible_family_roles[role]) < 8 for role in ROLES):
        raise RuntimeError(f"pre-forward family coverage below lock: {eligibility_audit}")
    write_json(output / "preforward_template_eligibility.json", eligibility_audit)
    heldout_panel: list[dict[str, Any]] = []
    for ref in sorted(heldout_core, key=lambda value: (value.depth, ROLES.index(value.role))):
        refs = heldout_refs_by_source[ref.source_key]
        panel_a = [v for v in refs if int(ref_hash(v, args.seed, "heldout_panel"), 16) % 2 == 0]
        panel_b = [v for v in refs if int(ref_hash(v, args.seed, "heldout_panel"), 16) % 2 == 1]
        if not panel_a or not panel_b:
            ordered = sorted(refs, key=lambda value: ref_hash(value, args.seed, "heldout_fallback"))
            panel_a, panel_b = ordered[::2], ordered[1::2]
        context = min(panel_a, key=lambda value: ref_hash(value, args.seed, "context_a"))
        score = min(panel_b, key=lambda value: ref_hash(value, args.seed, "score_b"))
        if (context.chunk_idx, context.record_idx) == (score.chunk_idx, score.record_idx):
            raise RuntimeError(f"heldout A/B refs coincide for {ref.source_key}")
        heldout_panel.append({"context": context, "score": score})
    index_audit = {
        "source": source_audit,
        "heldout": heldout_audit,
        "source_models": source_models,
        "source_structure": structure_audit,
        "c0_selected_records": len(c0_selection),
        "gain_selected_matrices": len(gain_refs),
        "heldout_core_matrices": len(heldout_panel),
        "heldout_record_counts": sorted({len(heldout_refs_by_source[ref.source_key]) for ref in heldout_core}),
        "heldout_panel_sha256": stable_hex(*[
            f"{item['context'].source_key}:{item['context'].chunk_idx}:{item['context'].record_idx}:"
            f"{item['score'].chunk_idx}:{item['score'].record_idx}" for item in heldout_panel
        ]),
    }
    write_json(output / "index_audit.json", index_audit)
    write_json(output / "c0_sampling_manifest.json", [
        {
            **{key: item[key] for key in (
                "model_name", "role", "native_ordinal", "native_layers", "native_u", "dataset",
                "eligible_for_cell_template", "available_distinct_datasets"
            )},
            "ref": item["ref"].as_dict(),
        }
        for item in c0_selection
    ])
    write_json(output / "gain_sampling_manifest.json", [ref.as_dict() for ref in gain_refs])
    write_json(output / "heldout_panel_manifest.json", [
        {"context_a": item["context"].as_dict(), "score_b": item["score"].as_dict()}
        for item in heldout_panel
    ])
    # This frozen snapshot is written before checkpoint/model forward. It binds
    # all refs, partitions, conditions, and staging decisions for this run.
    pre_forward = json.loads((output / "execution_manifest.json").read_text(encoding="utf-8"))
    pre_forward["runtime_lock"] = {
        "resolved_config_sha256": sha256_file(output / "resolved_config.json"),
        "index_audit_sha256": sha256_file(output / "index_audit.json"),
        "c0_sampling_manifest_sha256": sha256_file(output / "c0_sampling_manifest.json"),
        "gain_sampling_manifest_sha256": sha256_file(output / "gain_sampling_manifest.json"),
        "heldout_panel_manifest_sha256": sha256_file(output / "heldout_panel_manifest.json"),
        "preforward_template_eligibility_sha256": sha256_file(output / "preforward_template_eligibility.json"),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "status": "LOCKED_BEFORE_ANY_AE_OR_DISTRIBUTION_ENCODER_FORWARD",
    }
    write_json(output / "execution_manifest.json", pre_forward)
    logger.info("stage=pre_forward_lock artifact=%s", output / "execution_manifest.json")
    if args.audit_only:
        logger.info("completed audit_only=true; no checkpoint/model/distribution-encoder forward")
        return

    logger.info("stage=checkpoint_load")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["config"])
    model_cfg = _build_model_cfg(cfg)
    model_cfg.big_vae.use_latent_sampling = False
    model_cfg.big_vae.rope_2d_coord_kind = "raw"
    model = build_weight_quantile_vae(model_cfg).to(device)
    model.load_state_dict(_normalize_model_state_dict_keys(payload["model_state"]), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    amp_enabled, amp_dtype = _resolve_amp(cfg, device)
    amp_enabled = bool(amp_enabled and not args.no_amp)
    checkpoint_info = {
        "path": str(args.checkpoint.resolve()),
        "sha256": sha256_file(args.checkpoint),
        "step": int(payload.get("step", 0)),
        "stage": int(payload.get("stage", 0)),
        "rope_2d_coord_kind": str(model.cfg.big_vae.rope_2d_coord_kind),
        "use_latent_sampling": bool(model.cfg.big_vae.use_latent_sampling),
        "strict_load": True,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
    }
    if (
        checkpoint_info["sha256"] != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_info["step"] != EXPECTED_CHECKPOINT_STEP
        or checkpoint_info["rope_2d_coord_kind"] != "raw"
        or checkpoint_info["use_latent_sampling"] is not False
        or checkpoint_info["strict_load"] is not True
    ):
        raise RuntimeError(f"canonical checkpoint contract failed: {checkpoint_info}")
    write_json(output / "checkpoint_info.json", checkpoint_info)
    logger.info("checkpoint_info=%s", checkpoint_info)

    logger.info("stage=c0_extraction selected_records=%s", len(c0_selection))
    source_dataset = build_dataset(args.source_root, args.seed)
    c_vectors: list[dict[str, Any]] = []
    c0_io_order = sorted(c0_selection, key=lambda item: (item["ref"].chunk_idx, item["ref"].record_idx))
    for idx, item in enumerate(c0_io_order):
        _W, X = load_sample(source_dataset, item["ref"])
        c_var, c_patch, row_groups = extract_condition_vector(
            model=model,
            X=X,
            source_key=item["ref"].source_key,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        c_vectors.append({**item, "source_key": item["ref"].source_key, "c_var": c_var, "c_patch": c_patch})
        logger.info(
            "stage=c0_extraction record=%s/%s model=%s role=%s bin=%s row_groups=%s",
            idx + 1, len(c0_io_order), item["model_name"], item["role"],
            item["native_ordinal"], row_groups,
        )
    templates, medoid_manifest, hierarchy_manifest = build_templates(c_vectors)
    torch.save(templates, output / "source_condition_templates.pt")
    template_summary: list[dict[str, Any]] = []
    for kind, values in templates.items():
        entries = {("global", -1): values} if kind == "global" else values
        for (role, bin_idx), value in entries.items():
            template_summary.append({
                "kind": kind,
                "role": role,
                "canonical_depth": bin_idx,
                "c_var_sha256": tensor_sha256(value["c_var"]),
                "c_patch_sha256": tensor_sha256(value["c_patch"]),
                "c_var_norm": float(value["c_var"].norm()),
                "c_patch_norm": float(value["c_patch"].norm()),
            })
    write_csv(output / "condition_template_summary.csv", template_summary)
    write_json(output / "condition_medoid_manifest.json", medoid_manifest)
    write_json(output / "condition_hierarchy_manifest.json", hierarchy_manifest)

    logger.info("stage=heldout_load matrices=72")
    heldout_dataset = build_dataset(args.heldout_root, args.seed)
    heldout_records: list[MatrixRecord] = []
    for idx, item in enumerate(heldout_panel):
        W_a, X_a = load_sample(heldout_dataset, item["context"])
        W_b, X_b = load_sample(heldout_dataset, item["score"])
        if not torch.equal(W_a, W_b) or torch.equal(X_a, X_b):
            raise RuntimeError(f"heldout A/B contract failed: {item}")
        ref = item["context"]
        heldout_records.append(MatrixRecord(
            model_name=ref.model_name,
            depth=ref.depth,
            canonical_depth=ref.depth,
            role=ref.role,
            layer_name=ref.layer_name,
            source_key=ref.source_key,
            context_ref=ref,
            score_ref=item["score"],
            W=W_a,
            X_context=X_a,
            X_score=X_b,
        ))
        logger.info("stage=heldout_load matrix=%s/72 depth=%s role=%s shape=%s", idx+1, ref.depth, ref.role, tuple(W_a.shape))
    if {(record.depth, record.role) for record in heldout_records} != {(d, r) for d in range(12) for r in ROLES}:
        raise RuntimeError("heldout depth/role grid is not exact 12x6")

    logger.info("stage=g0_tiling_validation")
    coverage_rows: list[dict[str, Any]] = []
    for seed in tiling_seeds:
        for record in heldout_records:
            coverage_rows.append(validate_tiling(record, make_tiling(record, seed)))
    write_csv(output / "tiling_coverage.csv", coverage_rows)
    for seed in tiling_seeds:
        count = sum(row["num_tiles"] for row in coverage_rows if row["tiling_seed"] == seed)
        if count != 20736:
            raise RuntimeError(f"expected 20736 tiles for seed {seed}, got {count}")

    second_lock = json.loads((output / "execution_manifest.json").read_text(encoding="utf-8"))
    second_lock["template_tiling_lock"] = {
        "status": "LOCKED_TEMPLATES_AND_TILINGS_BEFORE_FIRST_WEIGHT_AE_FORWARD",
        "checkpoint_info_sha256": sha256_file(output / "checkpoint_info.json"),
        "source_condition_templates_sha256": sha256_file(output / "source_condition_templates.pt"),
        "condition_template_summary_sha256": sha256_file(output / "condition_template_summary.csv"),
        "condition_hierarchy_manifest_sha256": sha256_file(output / "condition_hierarchy_manifest.json"),
        "condition_medoid_manifest_sha256": sha256_file(output / "condition_medoid_manifest.json"),
        "tiling_coverage_sha256": sha256_file(output / "tiling_coverage.csv"),
        "tiling_partition_digest_sha256": stable_hex(
            *[str(row["partition_sha256"]) for row in coverage_rows]
        ),
        "template_tensor_cells": len(template_summary),
        "tiling_rows": len(coverage_rows),
        "tiles_per_tiling": {
            str(seed): sum(int(row["num_tiles"]) for row in coverage_rows if int(row["tiling_seed"]) == seed)
            for seed in tiling_seeds
        },
    }
    write_json(output / "execution_manifest.json", second_lock)
    logger.info("stage=template_tiling_lock artifact=%s", output / "execution_manifest.json")

    logger.info("stage=g0_repeat_and_dummy_invariance")
    probe = heldout_records[0]
    probe_tiling = make_tiling(probe, tiling_seeds[0])
    probe_W = split_tiles(probe.W, probe_tiling)[:2].to(device)
    template = resolve_template(templates, "ae_global_c0", probe.role, probe.canonical_depth)
    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        repeat_a = fixed_forward(model, probe_W, template)
        repeat_b = fixed_forward(model, probe_W, template)
        dummy_a = torch.zeros((2, 7, 64), device=device)
        dummy_b = torch.randn((2, 13, 64), device=device)
        # The fixed-C stub deliberately ignores both dummy tensors.
        dummy_out_a = fixed_forward_with_dummy_x(model, probe_W, template, dummy_a)
        dummy_out_b = fixed_forward_with_dummy_x(model, probe_W, template, dummy_b)
        native_X = torch.stack([probe.X_context[:, probe_tiling.rows[0]]] * 2).to(device)
        masks_x = torch.ones((2, native_X.shape[1]), dtype=torch.bool, device=device)
        masks_in = torch.ones((2, 64), dtype=torch.bool, device=device)
        masks_out = torch.ones((2, 64), dtype=torch.bool, device=device)
        native_a = model(probe_W, native_X, x_mask=masks_x, d_in_mask=masks_in, d_out_mask=masks_out)[0]
        native_b = model(probe_W, native_X, x_mask=masks_x, d_in_mask=masks_in, d_out_mask=masks_out)[0]
        native_manual = manual_native_forward(
            model,
            probe_W,
            native_X,
            x_mask=masks_x,
            d_in_mask=masks_in,
            d_out_mask=masks_out,
        )
    c_var_broadcast = template["c_var"].view(1, 1, 1, -1).expand(2, 4, 16, -1)
    c_patch_broadcast = template["c_patch"].view(1, 1, -1).expand(2, 4, -1)
    broadcast_identical = bool(
        torch.equal(c_var_broadcast[0, 0, 0], c_var_broadcast[-1, -1, -1])
        and torch.equal(c_patch_broadcast[0, 0], c_patch_broadcast[-1, -1])
    )
    derived_pooled = c_var_broadcast.mean(dim=2)
    pooled_reference = c_var_broadcast[:, :, 0, :]
    pooled_abs_error = (derived_pooled - pooled_reference).abs()
    pooled_reference_abs = pooled_reference.abs()
    pooled_rel_error = torch.where(
        pooled_reference_abs > 0,
        pooled_abs_error / pooled_reference_abs,
        torch.where(
            pooled_abs_error == 0,
            torch.zeros_like(pooled_abs_error),
            torch.full_like(pooled_abs_error, float("inf")),
        ),
    )
    pooled_bit_exact = bool(torch.equal(derived_pooled, pooled_reference))
    pooled_reduction_terms = int(c_var_broadcast.shape[2])
    pooled_finfo = torch.finfo(c_var_broadcast.dtype)
    # A mean of n identical floating-point values is mathematically the value
    # itself, but the finite-precision reduction need not be bit-exact.  Gate
    # against the standard sequential-summation relative error bound gamma_(n-1)
    # while retaining bit equality as a non-gating diagnostic.
    pooled_gamma = (
        (pooled_reduction_terms - 1) * float(pooled_finfo.eps)
        / (1.0 - (pooled_reduction_terms - 1) * float(pooled_finfo.eps))
    )
    pooled_semantic_parity = bool(
        torch.allclose(
            derived_pooled,
            pooled_reference,
            atol=float(pooled_finfo.tiny),
            rtol=pooled_gamma,
        )
    )
    invariance = {
        "fixed_repeat_bit_exact": bool(torch.equal(repeat_a, repeat_b)),
        "native_repeat_bit_exact": bool(torch.equal(native_a, native_b)),
        "native_public_vs_manual_bit_exact": bool(torch.equal(native_a, native_manual)),
        "native_public_vs_manual_allclose_atol_1e_6_rtol_1e_6": bool(
            torch.allclose(native_a, native_manual, atol=1e-6, rtol=1e-6)
        ),
        "native_public_vs_manual_max_abs": float((native_a.float() - native_manual.float()).abs().max()),
        "dummy_x_bit_exact": bool(torch.equal(dummy_out_a, dummy_out_b)),
        "dummy_shapes": [list(dummy_a.shape), list(dummy_b.shape)],
        "broadcast_c_var_and_c_patch_identical_across_tiles": broadcast_identical,
        "broadcast_c_var_sha256": tensor_sha256(c_var_broadcast[0, 0, 0]),
        "broadcast_c_patch_sha256": tensor_sha256(c_patch_broadcast[0, 0]),
        "c_pooled_matches_broadcast_c_var_mean_bit_exact_non_gating": pooled_bit_exact,
        "c_pooled_matches_broadcast_c_var_mean_within_reduction_roundoff": pooled_semantic_parity,
        "c_pooled_derivation": "broadcast_c_var.mean(dim=2)",
        "c_pooled_reduction_dtype": str(c_var_broadcast.dtype),
        "c_pooled_reduction_terms": pooled_reduction_terms,
        "c_pooled_reduction_dtype_eps": float(pooled_finfo.eps),
        "c_pooled_reduction_gamma_n_minus_1_rtol": pooled_gamma,
        "c_pooled_reduction_atol": float(pooled_finfo.tiny),
        "c_pooled_mismatch_elements": int((pooled_abs_error != 0).sum()),
        "c_pooled_total_elements": int(pooled_abs_error.numel()),
        "c_pooled_max_abs_error": float(pooled_abs_error.max()),
        "c_pooled_mean_abs_error": float(pooled_abs_error.mean()),
        "c_pooled_max_rel_error": float(pooled_rel_error.max()),
    }
    required_invariance_keys = (
        "fixed_repeat_bit_exact",
        "native_repeat_bit_exact",
        "native_public_vs_manual_allclose_atol_1e_6_rtol_1e_6",
        "dummy_x_bit_exact",
        "broadcast_c_var_and_c_patch_identical_across_tiles",
        "c_pooled_matches_broadcast_c_var_mean_within_reduction_roundoff",
    )
    # Persist every diagnostic even on a hard G0 failure so an abort remains
    # auditable instead of losing the exact failing values in terminal output.
    write_json(output / "forward_invariance.json", invariance)
    if not all(bool(invariance[key]) for key in required_invariance_keys):
        raise RuntimeError(f"G0 deterministic/invariance failure: {invariance}")

    logger.info("stage=source_gain_panel_load matrices=%s", len(gain_refs))
    gain_records: list[MatrixRecord] = []
    for idx, ref in enumerate(gain_refs):
        W, X = load_sample(source_dataset, ref)
        gain_records.append(MatrixRecord(
            model_name=ref.model_name,
            depth=ref.depth,
            canonical_depth=canonical_depth(*structure[ref.source_key]),
            role=ref.role,
            layer_name=ref.layer_name,
            source_key=ref.source_key,
            context_ref=ref,
            score_ref=ref,
            W=W,
            X_context=X,
            X_score=X,
        ))
        logger.info("stage=source_gain_panel_load matrix=%s/%s model=%s role=%s depth=%s", idx+1, len(gain_refs), ref.model_name, ref.role, ref.depth)
    gain_seed = args.seed * 100 + 3
    gain_stat_rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(gain_records):
        tiling = make_tiling(record, gain_seed)
        validate_tiling(record, tiling)
        for method in FIXED_METHODS:
            prediction = decode_matrix(
                model=model, record=record, tiling=tiling, method=method, templates=templates,
                device=device, batch_size=args.batch_size, amp_enabled=amp_enabled, amp_dtype=amp_dtype, logger=logger,
            )
            gain_stat_rows.append({"method": method, "role": record.role, **sufficient_stats(record.W, prediction, record.X_score)})
        logger.info("stage=gain_fit_prediction matrix=%s/%s", record_idx+1, len(gain_records))
    gains = fit_gains(gain_stat_rows)
    gain_rows = [
        {"method": method, "role": role, "gain": gains[(method, role)], "in_allowed_interval": 0.25 <= gains[(method, role)] <= 4.0}
        for method in FIXED_METHODS for role in ROLES
    ]
    write_csv(output / "source_fit_role_gains.csv", gain_rows)
    logger.info("source_role_gains=%s", gain_rows)
    del gain_records, gain_stat_rows

    logger.info("stage=heldout_evaluation")
    matrix_rows: list[dict[str, Any]] = []
    for tiling_seed in tiling_seeds:
        seed_tile_total = 0
        for record_idx, record in enumerate(heldout_records):
            tiling = make_tiling(record, tiling_seed)
            seed_tile_total += tiling.num_tiles
            predictions: dict[str, torch.Tensor] = {}
            for method in FIXED_METHODS + ("ae_native_c", "ae_zero_c"):
                predictions[method] = decode_matrix(
                    model=model, record=record, tiling=tiling, method=method, templates=templates,
                    device=device, batch_size=args.batch_size, amp_enabled=amp_enabled, amp_dtype=amp_dtype, logger=logger,
                )
            predictions["identity"] = record.W.clone()
            predictions["zero"] = torch.zeros_like(record.W)
            for method in METHODS:
                gain = gains[(method, record.role)] if method in FIXED_METHODS else 1.0
                row = metric_row(
                    record, tiling_seed=tiling_seed, method=method,
                    prediction=predictions[method], gain=gain, num_tiles=tiling.num_tiles,
                )
                if not all(math.isfinite(float(row[key])) for key in ("raw_E_W", "raw_E_X", "calibrated_E_W", "calibrated_E_X", "raw_weight_cosine", "raw_operator_cosine")):
                    # Zero has undefined cosine by construction and is handled below.
                    if method != "zero":
                        raise RuntimeError(f"non-finite metric: {row}")
                matrix_rows.append(row)
            logger.info("stage=heldout_evaluation seed=%s matrix=%s/72 depth=%s role=%s", tiling_seed, record_idx+1, record.depth, record.role)
        if seed_tile_total != 20736:
            raise RuntimeError(f"heldout tile total mismatch: seed={tiling_seed} total={seed_tile_total}")

    # Replace the undefined zero cosines with explicit NaN-free control values;
    # G1 finiteness applies to AE, while G0 zero checks concern only E_W/E_X.
    for row in matrix_rows:
        if row["method"] == "zero":
            row["raw_weight_cosine"] = 0.0
            row["raw_operator_cosine"] = 0.0
    aggregate = ratio_rows(matrix_rows)
    bootstrap = bootstrap_rows(matrix_rows, draws=args.bootstrap_draws, seed=args.seed + 99)
    write_csv(output / "matrix_metrics.csv", matrix_rows)
    write_csv(output / "aggregate_metrics.csv", aggregate)
    write_csv(output / "block_bootstrap.csv", bootstrap)

    def agg(seed: int, method: str, aggregation: str) -> dict[str, Any]:
        return next(row for row in aggregate if row["tiling_seed"] == seed and row["method"] == method and row["aggregation"] == aggregation)

    def boot(seed: int, method: str, metric: str, calibration: str = "calibrated") -> dict[str, Any]:
        return next(row for row in bootstrap if row["tiling_seed"] == seed and row["method"] == method and row["metric"] == metric and row["calibration"] == calibration)

    decisions: dict[str, Any] = {
        "tilings": {},
        "primary": "ae_cell_mean_c0",
        "global_c0_status": "mandatory stronger stress arm, not primary",
        "medoid_status": "support sensitivity; cannot rescue a failed mean",
    }
    all_g0 = True
    all_g1 = True
    for seed in tiling_seeds:
        identity_ok = all(agg(seed, "identity", scope)[metric] <= 1e-10 for scope in ("macro", "micro") for metric in ("raw_E_W", "raw_E_X"))
        zero_ok = all(0.999999 <= agg(seed, "zero", scope)[metric] <= 1.000001 for scope in ("macro", "micro") for metric in ("raw_E_W", "raw_E_X"))
        native_ok = all(agg(seed, "ae_native_c", scope)["raw_E_X"] < 1.0 for scope in ("macro", "micro"))
        role_points_raw = {role: agg(seed, "ae_cell_mean_c0", role)["raw_E_X"] for role in ROLES}
        role_points_calibrated = {role: agg(seed, "ae_cell_mean_c0", role)["calibrated_E_X"] for role in ROLES}
        primary_gain_ok = all(0.25 <= gains[("ae_cell_mean_c0", role)] <= 4.0 for role in ROLES)
        g1_checks = {
            "raw_macro_E_X_u95_below_0p90": boot(seed, "ae_cell_mean_c0", "macro_E_X", "raw")["u95"] < 0.90,
            "raw_micro_E_X_u95_below_0p90": boot(seed, "ae_cell_mean_c0", "micro_E_X", "raw")["u95"] < 0.90,
            "raw_macro_E_W_u95_below_1": boot(seed, "ae_cell_mean_c0", "macro_E_W", "raw")["u95"] < 1.0,
            "raw_micro_E_W_u95_below_1": boot(seed, "ae_cell_mean_c0", "micro_E_W", "raw")["u95"] < 1.0,
            "raw_every_role_E_X_below_1": all(value < 1.0 for value in role_points_raw.values()),
            "all_primary_role_gains_in_range": primary_gain_ok,
        }
        hostile_calibrated_checks = {
            "macro_E_X_u95_below_0p90": boot(seed, "ae_cell_mean_c0", "macro_E_X")["u95"] < 0.90,
            "micro_E_X_u95_below_0p90": boot(seed, "ae_cell_mean_c0", "micro_E_X")["u95"] < 0.90,
            "macro_E_W_u95_below_1": boot(seed, "ae_cell_mean_c0", "macro_E_W")["u95"] < 1.0,
            "micro_E_W_u95_below_1": boot(seed, "ae_cell_mean_c0", "micro_E_W")["u95"] < 1.0,
            "every_role_E_X_below_1": all(value < 1.0 for value in role_points_calibrated.values()),
        }
        g0_checks = {
            "identity_controls": identity_ok,
            "zero_controls": zero_ok,
            "native_macro_micro_below_zero": native_ok,
            "exact_72_matrices": True,
            "exact_20736_tiles": True,
            "coverage_and_reassembly": True,
            "repeat_and_dummy_invariance": all(
                bool(invariance[key]) for key in (
                    "fixed_repeat_bit_exact", "native_repeat_bit_exact",
                    "native_public_vs_manual_allclose_atol_1e_6_rtol_1e_6", "dummy_x_bit_exact",
                    "broadcast_c_var_and_c_patch_identical_across_tiles",
                    "c_pooled_matches_broadcast_c_var_mean_within_reduction_roundoff",
                )
            ),
        }
        all_g0 = all_g0 and all(g0_checks.values())
        all_g1 = all_g1 and all(g1_checks.values()) and all(hostile_calibrated_checks.values())
        diagnostics = {}
        for method in ("ae_global_c0", "ae_cell_medoid_c0", "ae_zero_c"):
            diagnostics[method] = {
                "raw_macro_E_X": agg(seed, method, "macro")["raw_E_X"],
                "raw_micro_E_X": agg(seed, method, "micro")["raw_E_X"],
                "macro_E_X": agg(seed, method, "macro")["calibrated_E_X"],
                "micro_E_X": agg(seed, method, "micro")["calibrated_E_X"],
                "macro_E_X_u95": boot(seed, method, "macro_E_X")["u95"],
                "micro_E_X_u95": boot(seed, method, "micro_E_X")["u95"],
                "every_role_E_X_below_1": all(agg(seed, method, role)["calibrated_E_X"] < 1.0 for role in ROLES),
            }
        decisions["tilings"][str(seed)] = {
            "G0": {"pass": all(g0_checks.values()), "checks": g0_checks},
            "G1_primary_cell_mean": {
                "pass": all(g1_checks.values()) and all(hostile_calibrated_checks.values()),
                "raw_primary_checks": g1_checks,
                "hostile_calibrated_checks": hostile_calibrated_checks,
                "raw_role_points": role_points_raw,
                "calibrated_role_points": role_points_calibrated,
                "raw_macro_E_X": agg(seed, "ae_cell_mean_c0", "macro")["raw_E_X"],
                "raw_micro_E_X": agg(seed, "ae_cell_mean_c0", "micro")["raw_E_X"],
                "raw_macro_E_X_u95": boot(seed, "ae_cell_mean_c0", "macro_E_X", "raw")["u95"],
                "raw_micro_E_X_u95": boot(seed, "ae_cell_mean_c0", "micro_E_X", "raw")["u95"],
                "calibrated_macro_E_X": agg(seed, "ae_cell_mean_c0", "macro")["calibrated_E_X"],
                "calibrated_micro_E_X": agg(seed, "ae_cell_mean_c0", "micro")["calibrated_E_X"],
            },
            "predeclared_nonprimary_diagnostics": diagnostics,
            "native_raw": {
                "macro_E_X": agg(seed, "ae_native_c", "macro")["raw_E_X"],
                "micro_E_X": agg(seed, "ae_native_c", "micro")["raw_E_X"],
            },
        }
    decisions["G0_pass_both_tilings"] = all_g0
    decisions["G1_primary_cell_mean_pass_both_tilings"] = all_g1
    decisions["action"] = (
        "ELIGIBLE_FOR_G2_SOURCE_CODECS; TARGETS_REMAIN_SEALED"
        if all_g0 and all_g1
        else "DO_NOT_RUN_G2_OR_UNSEAL_TARGET; DIAGNOSE_RECORDED_SOURCE FAILURE"
    )
    write_json(output / "decisions.json", decisions)
    plot_results(aggregate, bootstrap, output / "g01_operator_error.png")

    readme = "# Confirmatory held-out ViT-B source gate: G0/G1\n\n"
    readme += f"Checkpoint: `{checkpoint_info['sha256']}` at step `{checkpoint_info['step']}`, raw RoPE, deterministic AE.\n\n"
    readme += f"G0 pass on both locked tilings: **{all_g0}**.  Source-cell-mean C[role,depth] G1 pass on both: **{all_g1}**.\n\n"
    readme += "The 12-cell arithmetic source mean is the locked primary. Global C0 is the stronger stress arm; medoid is support sensitivity and cannot rescue a failed mean. No data2vec/target AE forward was performed.\n\n"
    readme += "See `decisions.json`, `aggregate_metrics.csv`, `block_bootstrap.csv`, `matrix_metrics.csv`, `tiling_coverage.csv`, and `g01_operator_error.png`.\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "status": "COMPLETE_G0_G1",
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": checkpoint_info,
        "counts": {
            "heldout_matrices": 72,
            "tiles_per_method_per_tiling": 20736,
            "matrix_metric_rows": len(matrix_rows),
            "bootstrap_draws": args.bootstrap_draws,
        },
        "decision": decisions,
        "artifacts": sorted(str(path) for path in output.iterdir()),
        "target_data2vec_access": False,
    }
    write_json(output / "run_manifest.json", manifest)
    logger.info("stage=output_writing artifacts=%s", manifest["artifacts"])
    logger.info("completed elapsed_s=%.1f G0=%s G1=%s action=%s", manifest["elapsed_seconds"], all_g0, all_g1, decisions["action"])


if __name__ == "__main__":
    main()
