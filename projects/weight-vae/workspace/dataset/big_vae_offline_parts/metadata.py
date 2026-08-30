from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from dataset.shared.types import SharedSample


OFFLINE_BIG_VAE_FORMAT_VERSION = 2
PRESLICED_BIG_VAE_FORMAT_VERSION = 1

_DEPTH_PATTERNS = (
    re.compile(r"(?:^|\.)(?:layers|layer|blocks|block|h|resblocks|encoder_layers|decoder_layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:encoder|decoder)\.(?:layers|layer|blocks|block)\.(\d+)(?:\.|$)"),
)

_BALANCED_SAMPLING_GROUP_KEY_ALIASES = {
    "dataset": "dataset",
    "model": "model",
    "layer_type": "layer_type",
    "depth": "depth",
    "shape": "shape",
    "source": "source",
    "layer": "layer",
    "d_in_bucket": "d_in_bucket",
    "d_out_bucket": "d_out_bucket",
    "num_params_bucket": "num_params_bucket",
}


def resolve_big_vae_curriculum_targets(cfg: DictConfig) -> tuple[int, int, int, int]:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    stage = max(1, int(train_cfg.get("stage", 1)))
    base_T = int(train_cfg.get("stage_base_T_patches", 4))
    base_d_out = int(train_cfg.get("stage_base_d_out", 16))
    scale = int(train_cfg.get("stage_scale_factor", 2))
    patch_size = max(1, int(model_cfg.get("patch_size", 16)))
    max_T_patches = base_T * (scale ** (stage - 1))
    max_d_out = base_d_out * (scale ** (stage - 1))
    max_x_rows = max(0, int(train_cfg.get("max_x_rows", 0)))
    return patch_size, max_T_patches, max_d_out, max_x_rows


def resolve_offline_target_size_bytes(offline_cfg: dict[str, Any] | DictConfig | None) -> int:
    cfg = dict(offline_cfg or {})
    raw_bytes = cfg.get("target_size_bytes", 0)
    if raw_bytes not in (None, ""):
        try:
            target_size_bytes = int(raw_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("train.offline_dataset.builder.target_size_bytes must be an integer") from exc
        if target_size_bytes > 0:
            return target_size_bytes

    raw_gb = cfg.get("target_size_gb", 0)
    if raw_gb not in (None, ""):
        try:
            target_size_gb = float(raw_gb)
        except (TypeError, ValueError) as exc:
            raise ValueError("train.offline_dataset.builder.target_size_gb must be a number") from exc
        if target_size_gb > 0.0:
            return int(target_size_gb * (1024.0 ** 3))

    raise ValueError(
        "Offline BigVAE dataset build requires a positive target size. "
        "Set train.offline_dataset.builder.target_size_bytes or target_size_gb."
    )


def estimate_stage_slice_capacity(
    *,
    d_in: int,
    d_out: int,
    patch_size: int,
    max_T_patches: int,
    max_d_out: int,
) -> int:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    total_patches = int(d_in) // int(patch_size)
    if total_patches <= 0 or int(d_out) <= 0:
        return 0

    used_patches = min(int(max_T_patches), total_patches)
    used_d_out = min(int(max_d_out), int(d_out))
    if used_patches <= 0 or used_d_out <= 0:
        return 0

    row_groups = 1 if total_patches <= used_patches else total_patches // used_patches
    col_groups = 1 if int(d_out) <= used_d_out else int(d_out) // used_d_out

    if total_patches <= used_patches and int(d_out) <= used_d_out:
        return 1
    return max(1, min(int(row_groups), int(col_groups)))


def _prepare_cpu_sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        return tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _sample_dataset_names(meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(meta, dict):
        return []
    image_meta = meta.get("image_meta", [])
    if not isinstance(image_meta, list):
        return []
    dataset_names = {
        str(item.get("dataset_name")).strip()
        for item in image_meta
        if isinstance(item, dict) and str(item.get("dataset_name", "")).strip()
    }
    return sorted(dataset_names)


def _primary_dataset_name(meta: dict[str, Any] | None) -> str:
    dataset_names = _sample_dataset_names(meta)
    return dataset_names[0] if dataset_names else "<unknown_dataset>"


def infer_layer_type(layer_name: str) -> str:
    name = str(layer_name).lower()

    if any(token in name for token in ("query", "q_proj", ".q.", "self.q", "attn.q", ".qkv")):
        return "attn_query"
    if any(token in name for token in ("key", "k_proj", ".k.", "self.k", "attn.k")):
        return "attn_key"
    if any(token in name for token in ("value", "v_proj", ".v.", "self.v", "attn.v")):
        return "attn_value"
    if any(token in name for token in ("out_proj", "output.dense", "attention.output", "attn.proj")):
        return "attn_output"
    if "attn" in name or "attention" in name:
        return "attn_other"

    if any(token in name for token in ("intermediate", "fc1", "mlp.fc1", "gate_proj", "up_proj")):
        return "ffn_up"
    if any(token in name for token in ("output.dense", "fc2", "mlp.fc2", "down_proj")):
        return "ffn_down"

    if "pooler" in name:
        return "pooler"
    if any(token in name for token in ("embed", "embedding")):
        return "embedding"
    if "conv" in name:
        return "conv"
    if any(token in name for token in ("lm_head", "classifier", "score", "head")):
        return "head"
    return "other_linear"


def infer_layer_depth(layer_name: str) -> int | None:
    name = str(layer_name).strip()
    if not name:
        return None
    for pattern in _DEPTH_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            return int(match.group(1))
    return None


def _shape_key(d_in: int, d_out: int) -> str:
    return f"{int(d_in)}x{int(d_out)}"


def _pow2_bucket(value: int) -> str:
    value = int(value)
    if value <= 0:
        return "0"
    lower = 1 << max(0, int(value).bit_length() - 1)
    upper = max(lower, (lower << 1) - 1)
    return f"{lower}-{upper}"


def _source_key(model_name: str, layer_name: str) -> str:
    payload = f"{str(model_name).strip()}\n{str(layer_name).strip()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

__all__ = [
    'OFFLINE_BIG_VAE_FORMAT_VERSION',
    'PRESLICED_BIG_VAE_FORMAT_VERSION',
    '_DEPTH_PATTERNS',
    '_BALANCED_SAMPLING_GROUP_KEY_ALIASES',
    'resolve_big_vae_curriculum_targets',
    'resolve_offline_target_size_bytes',
    'estimate_stage_slice_capacity',
    '_prepare_cpu_sample_tensor',
    '_sample_dataset_names',
    '_primary_dataset_name',
    'infer_layer_type',
    'infer_layer_depth',
    '_shape_key',
    '_pow2_bucket',
    '_source_key',
]
