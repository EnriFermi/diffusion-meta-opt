from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset


def setup_logging(cfg: DictConfig) -> None:
    level_name = str(cfg.data.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("train")

    logger.info("Starting training entrypoint")
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    try:
        if bool(cfg.train.get("predownload_models", False)):
            logger.info("Predownloading model artifacts before collector start")
            collector.predownload_models()

        collector.start()

        # Dataset is ready for the future training loop.
        dataset_iter = iter(dataset)
        _ = dataset_iter  # keep explicit reference for readability in the scaffold

        logger.info(
            "Data pipeline initialized: collector_mode=%s streaming_mode=%s cache_metric=%s",
            collector.collector_mode,
            collector.streaming_mode,
            dataset.cache_size(),
        )

        # Training loop will be implemented next.
        pass
    finally:
        dataset.close()
        collector.shutdown()
        logger.info("Training entrypoint shutdown complete")


if __name__ == "__main__":
    main()
