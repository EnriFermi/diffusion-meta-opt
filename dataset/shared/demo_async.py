from __future__ import annotations

import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import data_pipeline, setup_logging


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_async")

    logger.info("Starting async collector demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    demo_cfg = cfg.get("demo", {})
    runtime_seconds = int(demo_cfg.get("runtime_seconds", 10))

    with data_pipeline(cfg, logger=logger, emit_run_report=False) as (_, collector):
        if not collector.is_async_mode:
            raise ValueError("demo_async requires async collector mode (collector.device != train.device)")
        started = time.time()
        while time.time() - started < runtime_seconds:
            logger.info("cache size=%s stats=%s", collector.cache_size(), collector.stats())
            time.sleep(1.0)


if __name__ == "__main__":
    main()
