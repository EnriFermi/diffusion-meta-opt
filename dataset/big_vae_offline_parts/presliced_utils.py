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

from dataset.big_vae_offline_parts.metadata import *
from dataset.big_vae_offline_parts.chunk_io import *
from dataset.big_vae_offline_parts.offline_dataset import *

def _preslicing_cfg(cfg: DictConfig) -> dict[str, Any] | DictConfig:
    train_cfg = cfg.get("train", {})
    preslicing_cfg = train_cfg.get("preslicing", {})
    if preslicing_cfg is None:
        preslicing_cfg = {}
    if not isinstance(preslicing_cfg, (dict, DictConfig)):
        raise TypeError("train.preslicing must be a mapping")
    return preslicing_cfg


def _preslicing_source_root(cfg: DictConfig) -> str:
    offline_cfg = cfg.train.get("offline_dataset", {})
    if offline_cfg is None:
        offline_cfg = {}
    if not isinstance(offline_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset must be a mapping")
    source_root = str(offline_cfg.get("root_dir", "") or "").strip()
    if not source_root:
        raise ValueError("train.offline_dataset.root_dir must be set when train.preslicing.enabled=true")
    return source_root


def _preslicing_root(cfg: DictConfig) -> str:
    preslicing_cfg = _preslicing_cfg(cfg)
    root_dir = str(preslicing_cfg.get("root_dir", "") or "").strip()
    if root_dir:
        return root_dir
    source_root = Path(_preslicing_source_root(cfg))
    stage = max(1, int(cfg.train.get("stage", 1)))
    return str(source_root / "presliced" / f"stage_{stage}")


def _file_sha1(path: Path, *, logger: logging.Logger | None = None, label: str | None = None) -> str:
    digest = hashlib.sha1()
    size_bytes = int(path.stat().st_size)
    display_label = label or path.name
    if logger is not None:
        logger.info(
            "Hashing offline dataset fingerprint file: label=%s path=%s size_mb=%.1f",
            display_label,
            path,
            float(size_bytes) / (1024.0 ** 2),
        )
    read_bytes = 0
    last_log_t = time.perf_counter()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read_bytes += len(chunk)
            now = time.perf_counter()
            if logger is not None and size_bytes >= 512 * 1024 * 1024 and now - last_log_t >= 10.0:
                logger.info(
                    "Hashing offline dataset fingerprint file progress: label=%s read_mb=%.1f/%.1f",
                    display_label,
                    float(read_bytes) / (1024.0 ** 2),
                    float(size_bytes) / (1024.0 ** 2),
                )
                last_log_t = now
    return digest.hexdigest()


def _source_offline_dataset_fingerprint(
    source_root: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    root = Path(source_root)
    manifest_path = root / "manifest.json"
    sources_path = root / "sources.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source offline BigVAE manifest not found: {manifest_path}")
    manifest = _load_json_payload(manifest_path)
    payload = {
        "root_dir": str(root),
        "manifest_sha1": _file_sha1(manifest_path, logger=logger, label="manifest"),
        "sources_sha1": _file_sha1(sources_path, logger=logger, label="sources") if sources_path.exists() else "",
        "format_version": int(manifest.get("format_version", 0)),
        "accepted_records": int(manifest.get("accepted_records", 0)),
        "unique_sources": int(manifest.get("unique_sources", 0)),
        "actual_size_bytes": int(manifest.get("actual_size_bytes", 0)),
    }
    return payload


def resolve_presliced_big_vae_spec(
    cfg: DictConfig,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    preslicing_cfg = _preslicing_cfg(cfg)
    patch_size, max_T_patches, max_d_out, max_x_rows = resolve_big_vae_curriculum_targets(cfg)
    source_root = _preslicing_source_root(cfg)
    root_dir = _preslicing_root(cfg)

    num_slices = int(preslicing_cfg.get("num_slices", 0))
    if num_slices <= 0:
        raise ValueError("train.preslicing.num_slices must be > 0 when train.preslicing.enabled=true")

    raw_target_x_rows = int(preslicing_cfg.get("target_x_rows", 0))
    target_x_rows = raw_target_x_rows if raw_target_x_rows > 0 else int(max_x_rows)
    if target_x_rows <= 0:
        raise ValueError(
            "train.preslicing.target_x_rows must be > 0 when train.max_x_rows is not set or is <= 0"
        )

    raw_target_d_in = int(preslicing_cfg.get("target_d_in", 0))
    raw_target_d_out = int(preslicing_cfg.get("target_d_out", 0))
    target_d_in = raw_target_d_in if raw_target_d_in > 0 else int(patch_size) * int(max_T_patches)
    target_d_out = raw_target_d_out if raw_target_d_out > 0 else int(max_d_out)
    if target_d_in <= 0 or target_d_out <= 0:
        raise ValueError("train.preslicing target_d_in/target_d_out must be > 0")
    if target_d_in < int(patch_size):
        raise ValueError("train.preslicing.target_d_in must be at least model.patch_size")

    offline_cfg = cfg.train.get("offline_dataset", {})
    sampling_cfg = offline_cfg.get("sampling", {}) if isinstance(offline_cfg, (dict, DictConfig)) else {}
    spec = {
        "format_version": PRESLICED_BIG_VAE_FORMAT_VERSION,
        "root_dir": str(root_dir),
        "source_root_dir": str(source_root),
        "source_dataset": _source_offline_dataset_fingerprint(source_root, logger=logger),
        "num_slices": int(num_slices),
        "stage": max(1, int(cfg.train.get("stage", 1))),
        "patch_size": int(patch_size),
        "max_T_patches": int(max_T_patches),
        "max_d_out": int(max_d_out),
        "max_x_rows": int(max_x_rows),
        "target_x_rows": int(target_x_rows),
        "target_d_in": int(target_d_in),
        "target_d_out": int(target_d_out),
        "seed": int(preslicing_cfg.get("seed", cfg.data.get("seed", 42))),
        "source_sampling_mode": str(sampling_cfg.get("mode", "random")) if isinstance(sampling_cfg, (dict, DictConfig)) else "random",
        "source_sampling_group_keys": list(sampling_cfg.get("group_keys", ("dataset", "model", "layer_type", "depth")))
        if isinstance(sampling_cfg, (dict, DictConfig))
        else ["dataset", "model", "layer_type", "depth"],
    }
    fingerprint_payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    spec["fingerprint"] = hashlib.sha1(fingerprint_payload).hexdigest()
    return spec


def _presliced_manifest_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / "manifest.json"


def _presliced_manifest_matches(root_dir: str | Path, spec: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str]:
    manifest_path = _presliced_manifest_path(root_dir)
    if not manifest_path.exists():
        return False, None, "missing_manifest"
    try:
        manifest = _load_json_payload(manifest_path)
    except Exception as exc:
        return False, None, f"manifest_load_failed:{exc}"
    if int(manifest.get("format_version", 0)) != PRESLICED_BIG_VAE_FORMAT_VERSION:
        return False, manifest, "format_version_mismatch"
    if str(manifest.get("fingerprint", "")) != str(spec.get("fingerprint", "")):
        return False, manifest, "fingerprint_mismatch"
    if int(manifest.get("num_slices", 0)) != int(spec.get("num_slices", 0)):
        return False, manifest, "num_slices_mismatch"
    chunks_dir = Path(root_dir) / "chunks"
    if not chunks_dir.exists() or not any(chunks_dir.glob("*.pt")):
        return False, manifest, "missing_chunks"
    return True, manifest, "ok"


def _presliced_pad_x(x: torch.Tensor, target_rows: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = _prepare_cpu_sample_tensor(x)
    if int(x.shape[0]) > int(target_rows):
        keep = torch.randperm(int(x.shape[0]), generator=rng)[: int(target_rows)]
        keep = keep.sort().values
        x = x[keep]
    mask = torch.zeros((int(target_rows),), dtype=torch.bool)
    rows = int(x.shape[0])
    mask[:rows] = True
    if rows == int(target_rows):
        return x, mask
    pad = torch.zeros((int(target_rows) - rows, int(x.shape[1])), dtype=x.dtype)
    return torch.cat([x, pad], dim=0), mask


def _presliced_pad_matrix(
    W: torch.Tensor,
    x: torch.Tensor,
    *,
    target_d_in: int,
    target_d_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    W = _prepare_cpu_sample_tensor(W)
    x = _prepare_cpu_sample_tensor(x)
    d_in = int(W.shape[0])
    d_out = int(W.shape[1])
    if int(x.shape[1]) != d_in:
        raise ValueError(f"x last dim ({int(x.shape[1])}) must match W d_in ({d_in})")
    if d_in > int(target_d_in):
        raise ValueError(f"presliced d_in={d_in} exceeds target_d_in={target_d_in}")
    if d_out > int(target_d_out):
        raise ValueError(f"presliced d_out={d_out} exceeds target_d_out={target_d_out}")

    d_in_mask = torch.zeros((int(target_d_in),), dtype=torch.bool)
    d_out_mask = torch.zeros((int(target_d_out),), dtype=torch.bool)
    d_in_mask[:d_in] = True
    d_out_mask[:d_out] = True
    if d_in < int(target_d_in):
        x_pad = torch.zeros((int(x.shape[0]), int(target_d_in) - d_in), dtype=x.dtype)
        W_pad = torch.zeros((int(target_d_in) - d_in, d_out), dtype=W.dtype)
        x = torch.cat([x, x_pad], dim=1)
        W = torch.cat([W, W_pad], dim=0)
    if d_out < int(target_d_out):
        W_pad = torch.zeros((int(target_d_in), int(target_d_out) - d_out), dtype=W.dtype)
        W = torch.cat([W, W_pad], dim=1)
    return W.contiguous(), x.contiguous(), d_in_mask, d_out_mask


def _presliced_index_groups(
    *,
    num_items: int,
    group_size: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, ...] | None:
    if num_items <= 0 or group_size <= 0:
        return None
    if group_size >= num_items:
        return None
    num_groups = int(num_items) // int(group_size)
    if num_groups <= 0:
        return None
    order = torch.randperm(int(num_items), generator=rng)
    groups: list[torch.Tensor] = []
    for group_idx in range(num_groups):
        group = order[group_idx * int(group_size) : (group_idx + 1) * int(group_size)].sort().values
        groups.append(group)
    return tuple(groups)

__all__ = [
    '_preslicing_cfg',
    '_preslicing_source_root',
    '_preslicing_root',
    '_file_sha1',
    '_source_offline_dataset_fingerprint',
    'resolve_presliced_big_vae_spec',
    '_presliced_manifest_path',
    '_presliced_manifest_matches',
    '_presliced_pad_x',
    '_presliced_pad_matrix',
    '_presliced_index_groups',
]
