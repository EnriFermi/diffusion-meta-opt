from __future__ import annotations

import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import data_pipeline, setup_logging

# Demo-only runtime knobs.
# These values are intentionally local to this demo script and are NOT part of
# the core data-pipeline Hydra runtime schema (train/model/data configs).
DEMO_RUNTIME_SECONDS = 10


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_async")

    logger.info("Starting async collector demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _log_demo_settings(logger, {"runtime_seconds": DEMO_RUNTIME_SECONDS})

    with data_pipeline(cfg, logger=logger, emit_run_report=False) as (_, collector):
        if not collector.is_async_mode:
            raise ValueError("demo_async requires async collector mode (collector.device != train.device)")
        started = time.time()
        while time.time() - started < DEMO_RUNTIME_SECONDS:
            logger.info("cache size=%s stats=%s", collector.cache_size(), collector.stats())
            time.sleep(1.0)


def _log_demo_settings(logger: logging.Logger, values: dict[str, int]) -> None:
    logger.info("Demo-only settings (not part of core runtime config):")
    logger.info("+---------------------+--------+")
    logger.info("| parameter           | value  |")
    logger.info("+---------------------+--------+")
    for key, value in values.items():
        logger.info("| %-19s | %-6s |", key, value)
    logger.info("+---------------------+--------+")


if __name__ == "__main__":
    main()
