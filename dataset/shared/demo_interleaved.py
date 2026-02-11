from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_interleaved")

    logger.info("Starting interleaved collector demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    collector = CollectorService(cfg)
    if collector.is_async_mode:
        raise ValueError("demo_interleaved requires interleaved mode")

    demo_cfg = cfg.get("demo", {})
    steps = int(demo_cfg.get("steps", 120))

    collector.predownload_models()
    collector.start()
    dataset = SharedModelDataset(collector)

    try:
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
    finally:
        dataset.close()
        collector.shutdown()


def _setup_logging(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    level_name = str(data_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


if __name__ == "__main__":
    main()
