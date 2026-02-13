from __future__ import annotations

import contextlib
import logging
from typing import Iterator

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset


def setup_logging(cfg: DictConfig) -> None:
    level_name = str(cfg.data.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@contextlib.contextmanager
def data_pipeline(cfg: DictConfig) -> Iterator[tuple[SharedModelDataset, CollectorService]]:
    _logger = logging.getLogger("train")

    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    if bool(cfg.train.get("predownload_models", False)):
        _logger.info("Predownloading model artifacts before collector start")
        collector.predownload_models()

    collector.start()

    _logger.info(
        "Data pipeline initialized: collector_mode=%s streaming_mode=%s cache_metric=%s",
        collector.collector_mode,
        collector.streaming_mode,
        dataset.cache_size(),
    )

    try:
        yield dataset, collector
    finally:
        dataset.close()
        collector.shutdown()
        _logger.info("Training entrypoint shutdown complete")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("train")

    logger.info("Starting training entrypoint")
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    with data_pipeline(cfg) as (dataset, collector):
        # Dataset is ready for the future training loop.
        dataset_iter = iter(dataset)
        _ = dataset_iter  # keep explicit reference for readability in the scaffold

        # Training loop will be implemented next.
        pass


if __name__ == "__main__":
    main()
