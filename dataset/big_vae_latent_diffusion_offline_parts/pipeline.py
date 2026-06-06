from __future__ import annotations

import contextlib
import logging
from typing import Iterator

from omegaconf import DictConfig

from .dataset import OfflineBigVAELatentDiffusionDataset

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
