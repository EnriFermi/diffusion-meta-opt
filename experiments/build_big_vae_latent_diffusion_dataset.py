from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset.big_vae_offline import OfflineBigVAEDataset
from dataset.big_vae_latent_diffusion_offline import build_big_vae_latent_diffusion_offline_dataset
from dataset.logging_utils import configure_process_logging
from training.big_vae_latent_diffusion import load_frozen_big_vae_from_checkpoint


LOGGER = logging.getLogger("build_big_vae_latent_diffusion_dataset")


def _promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return
    expected_sections = ("latent_diffusion_dataset", "latent_diffusion_prior", "training_artifacts", "logging", "hf")
    with open_dict(cfg):
        for section in expected_sections:
            if section not in cfg and section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]


@hydra.main(version_base=None, config_path="../conf", config_name="config_big_vae_latent_diffusion_dataset_build")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    log_path = configure_process_logging(cfg=cfg, role="build_big_vae_latent_diffusion_dataset", rank=0, force=True)
    logger = LOGGER
    logger.info("Starting BigVAE latent diffusion dataset build")
    logger.info("Run log file: %s", log_path)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    dataset_cfg = cfg.get("latent_diffusion_dataset", {})
    if not isinstance(dataset_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset must be a mapping")
    source_cfg = dataset_cfg.get("source", {})
    if not isinstance(source_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset.source must be a mapping")
    prior_cfg = cfg.get("latent_diffusion_prior", {})
    if not isinstance(prior_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_prior must be a mapping")

    source_root = str(dataset_cfg.get("source_big_vae_offline_root", "") or "").strip()
    if not source_root:
        raise ValueError("latent_diffusion_dataset.source_big_vae_offline_root must be set")
    checkpoint_path = str(prior_cfg.get("big_vae_checkpoint", "") or "").strip()
    if not checkpoint_path:
        raise ValueError("latent_diffusion_prior.big_vae_checkpoint must be set")

    device_raw = str(dataset_cfg.get("device", "auto")).strip().lower()
    if device_raw == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_raw)
    logger.info("Loading frozen BigVAE checkpoint: %s", checkpoint_path)
    big_vae = load_frozen_big_vae_from_checkpoint(checkpoint_path, device=device)

    source_dataset = OfflineBigVAEDataset(
        root_dir=source_root,
        shuffle_chunks=bool(source_cfg.get("shuffle_chunks", False)),
        shuffle_records_within_chunk=bool(source_cfg.get("shuffle_records_within_chunk", False)),
        repeat=False,
        seed=int(source_cfg.get("seed", 42)),
        shard_rank=0,
        shard_world_size=1,
        shard_by_chunk=False,
        weight_cache_size=int(source_cfg.get("weight_cache_size", 64)),
        sampling_mode=str(source_cfg.get("sampling_mode", "random")),
        sampling_group_keys=source_cfg.get("sampling_group_keys", ("dataset", "model", "layer_type", "depth")),
        sampling_window_size=int(source_cfg.get("sampling_window_size", 2048)),
        sampling_max_records_per_chunk_round=int(source_cfg.get("sampling_max_records_per_chunk_round", 8)),
        x_chunk_cache_size=int(source_cfg.get("x_chunk_cache_size", 4)),
    )

    try:
        summary = build_big_vae_latent_diffusion_offline_dataset(
            cfg,
            big_vae=big_vae,
            dataset_iter=source_dataset,
            logger=logger,
        )
    finally:
        source_dataset.close()

    logger.info(
        "Latent diffusion dataset build complete: root=%s accepted_records=%s actual_size_gb=%.2f",
        summary.get("root_dir", ""),
        int(summary.get("accepted_records", 0)),
        float(summary.get("actual_size_gb", 0.0)),
    )


if __name__ == "__main__":
    main()
