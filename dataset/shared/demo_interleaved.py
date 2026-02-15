from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import data_pipeline, setup_logging


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_interleaved")

    logger.info("Starting interleaved collector demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    demo_cfg = cfg.get("demo", {})
    steps = int(demo_cfg.get("steps", 120))

    with data_pipeline(cfg, logger=logger, emit_run_report=False) as (dataset, collector):
        if collector.is_async_mode:
            raise ValueError("demo_interleaved requires interleaved mode")
        for step_idx in range(steps):
            stats = collector.maybe_collect(step_idx)
            consumed = dataset.try_next_sample()
            logger.info(
                "step=%s cache=%s collected_jobs=%s consumed=%s",
                step_idx,
                collector.cache_size(),
                len(stats),
                consumed is not None,
            )


if __name__ == "__main__":
    main()
