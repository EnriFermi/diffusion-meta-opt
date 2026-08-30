from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import data_pipeline, setup_logging

# Demo-only runtime knobs.
# These values are intentionally local to this demo script and are NOT part of
# the core data-pipeline Hydra runtime schema (train/model/data configs).
DEMO_STEPS = 120


@hydra.main(version_base=None, config_path="../../conf", config_name="big_vae/train/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_interleaved")

    logger.info("Starting interleaved collector demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _log_demo_settings(logger, {"steps": DEMO_STEPS})

    with data_pipeline(cfg, logger=logger, emit_run_report=False) as (dataset, collector):
        if collector.is_async_mode:
            raise ValueError("demo_interleaved requires interleaved mode")
        for step_idx in range(DEMO_STEPS):
            stats = collector.maybe_collect(step_idx)
            consumed = dataset.try_next_sample()
            logger.info(
                "step=%s cache=%s collected_jobs=%s consumed=%s",
                step_idx,
                collector.cache_size(),
                len(stats),
                consumed is not None,
            )


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
