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
from dataset.big_vae_offline_parts.presliced_utils import *
from dataset.big_vae_offline_parts.presliced_dataset import *

def build_big_vae_offline_dataset(
    cfg: DictConfig,
    *,
    dataset_iter: Iterator[Any],
    logger: logging.Logger | None = None,
    max_seen_samples: int = 0,
) -> dict[str, Any]:
    offline_cfg = cfg.train.get("offline_dataset", {})
    if offline_cfg is None:
        offline_cfg = {}
    if not isinstance(offline_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset must be a mapping")
    builder_cfg = offline_cfg.get("builder", {})
    if builder_cfg is None:
        builder_cfg = {}
    if not isinstance(builder_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset.builder must be a mapping")

    patch_size, max_T_patches, max_d_out, max_x_rows = resolve_big_vae_curriculum_targets(cfg)
    target_size_bytes = resolve_offline_target_size_bytes(builder_cfg)
    logger_local = logger or logging.getLogger("dataset.big_vae_offline")
    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    root_dir = str(offline_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("train.offline_dataset.root_dir must be set when building an offline BigVAE dataset")
    writer = BigVAEOfflineDatasetWriter(
        root_dir=root_dir,
        target_size_bytes=target_size_bytes,
        patch_size=patch_size,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
        max_x_rows=max_x_rows,
        x_chunk_size_records=int(builder_cfg.get("x_chunk_size_records", 128)),
        max_samples_per_source=int(builder_cfg.get("max_samples_per_source", 0)),
        enforce_stage_compatibility=bool(builder_cfg.get("enforce_stage_compatibility", False)),
        overwrite_existing=bool(builder_cfg.get("overwrite_existing", False)),
        seed=int(cfg.data.get("seed", 42)),
        logger=logger_local,
        config_snapshot=cfg_snapshot if isinstance(cfg_snapshot, dict) else {},
    )

    log_every_seen = max(1, int(builder_cfg.get("log_every_seen_samples", 1000)))
    seen_limit = max(0, int(max_seen_samples or builder_cfg.get("max_seen_samples", 0)))

    try:
        while not writer.reached_target_size:
            if seen_limit > 0 and int(writer.stats()["seen_samples"]) >= seen_limit:
                break
            try:
                sample = next(dataset_iter)
            except StopIteration:
                logger_local.warning(
                    "Offline BigVAE build stopped because dataset iterator was exhausted before the target size was reached"
                )
                break
            writer.ingest(sample)
            stats = writer.stats()
            if int(stats["seen_samples"]) % log_every_seen == 0:
                logger_local.info(
                    "Offline BigVAE build progress: seen=%s accepted=%s size_gb=%.2f/%s unique_sources=%s "
                    "skipped_invalid=%s skipped_incompatible=%s skipped_source_cap=%s stage_filter=%s",
                    int(stats["seen_samples"]),
                    int(stats["accepted_records"]),
                    float(stats["written_size_gb"]),
                    f"{float(target_size_bytes) / (1024.0 ** 3):.2f}",
                    int(stats["unique_sources"]),
                    int(stats["skipped_invalid"]),
                    int(stats["skipped_incompatible"]),
                    int(stats["skipped_source_cap"]),
                    bool(stats["enforce_stage_compatibility"]),
                )
    finally:
        summary = writer.close()

    if not writer.reached_target_size and seen_limit > 0:
        logger_local.warning(
            "Offline BigVAE build stopped before target size because max_seen_samples=%s was reached. "
            "actual_size_gb=%.2f target_size_gb=%.2f",
            seen_limit,
            float(summary.get("actual_size_gb", 0.0)),
            float(summary.get("target_size_bytes", 0.0)) / (1024.0 ** 3),
        )
    return summary

__all__ = [
    'build_big_vae_offline_dataset',
]
