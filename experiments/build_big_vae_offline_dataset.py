from __future__ import annotations

import logging
from typing import Any, Iterator

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset import data_pipeline
from dataset.big_vae_offline import (
    build_big_vae_offline_dataset,
    resolve_big_vae_curriculum_targets,
    resolve_offline_target_size_bytes,
)
from dataset.logging_utils import configure_process_logging


def _promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return

    expected_sections = (
        "data",
        "collector",
        "streaming",
        "train",
        "model",
        "training_artifacts",
        "logging",
        "hf",
        "models",
    )
    with open_dict(cfg):
        for section in expected_sections:
            if section in cfg:
                continue
            if section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]


def _dataset_iterator_with_collection(dataset: Any, collector: Any) -> Iterator[Any]:
    dataset_iter = iter(dataset)
    seen = 0
    while True:
        if collector is not None and not bool(getattr(collector, "is_async_mode", False)):
            dataset.maybe_collect(seen)
        yield next(dataset_iter)
        seen += 1


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    log_path = configure_process_logging(cfg=cfg, role="build_big_vae_offline_dataset", rank=0, force=True)
    logger = logging.getLogger("build_big_vae_offline_dataset")

    patch_size, max_T_patches, max_d_out, max_x_rows = resolve_big_vae_curriculum_targets(cfg)
    offline_cfg = cfg.train.get("offline_dataset", {})
    builder_cfg = offline_cfg.get("builder", {}) if isinstance(offline_cfg, (dict, DictConfig)) else {}
    target_size_bytes = resolve_offline_target_size_bytes(builder_cfg)

    logger.info("Starting BigVAE offline dataset build")
    logger.info("Run log file: %s", log_path)
    logger.info(
        "Build target: root=%s target_size_gb=%.2f patch_size=%s max_T_patches=%s max_d_out=%s max_x_rows=%s "
        "enforce_stage_compatibility=%s",
        str(offline_cfg.get("root_dir", "")),
        float(target_size_bytes) / (1024.0 ** 3),
        patch_size,
        max_T_patches,
        max_d_out,
        max_x_rows,
        bool(builder_cfg.get("enforce_stage_compatibility", False)),
    )
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    with data_pipeline(cfg, logger=logger, emit_run_report=True, rank=0) as (dataset, collector):
        summary = build_big_vae_offline_dataset(
            cfg,
            dataset_iter=_dataset_iterator_with_collection(dataset, collector),
            logger=logger,
        )

    logger.info(
        "Offline BigVAE dataset build complete: root=%s actual_size_gb=%.2f accepted_records=%s unique_sources=%s",
        summary.get("root_dir", str(offline_cfg.get("root_dir", ""))),
        float(summary.get("actual_size_gb", 0.0)),
        int(summary.get("accepted_records", 0)),
        int(summary.get("unique_sources", 0)),
    )


if __name__ == "__main__":
    main()
