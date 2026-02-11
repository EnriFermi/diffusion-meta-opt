from __future__ import annotations

import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset.shared.collector_service import CollectorService


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_async")

    logger.info("Starting async collector demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    collector = CollectorService(cfg)
    if not collector.is_async_mode:
        raise ValueError("demo_async requires async collector mode (collector.device != train.device)")

    demo_cfg = cfg.get("demo", {})
    runtime_seconds = int(demo_cfg.get("runtime_seconds", 10))

    collector.predownload_models()
    collector.start()

    try:
        started = time.time()
        while time.time() - started < runtime_seconds:
            logger.info("cache size=%s stats=%s", collector.cache.size(), collector.stats())
            time.sleep(1.0)
    finally:
        collector.shutdown()


def _setup_logging(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    level_name = str(data_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


if __name__ == "__main__":
    main()
