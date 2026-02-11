from __future__ import annotations

import logging
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_streaming_s3_bridge")

    logger.info("Starting S3-bridge streaming demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if str(cfg.streaming.mode).lower() != "s3_bridge":
        raise ValueError("demo_streaming_s3_bridge requires streaming.mode=s3_bridge")
    if not str(cfg.streaming.s3.bucket).strip():
        raise ValueError("streaming.s3.bucket must be set for demo_streaming_s3_bridge")

    demo_cfg = cfg.get("demo", {})
    num_samples = int(demo_cfg.get("num_samples", 200))

    collector = CollectorService(cfg)
    collector.predownload_models()
    collector.start()

    dataset = SharedModelDataset(collector)
    iterator = iter(dataset)
    model_counts: Counter[str] = Counter()

    try:
        step_idx = 0
        while sum(model_counts.values()) < num_samples:
            if not collector.is_async_mode:
                dataset.maybe_collect(step_idx)

            sample = next(iterator)
            model_counts[sample.model_name] += 1

            if sum(model_counts.values()) % 20 == 0:
                logger.info(
                    "consumed=%s remote_ready=%s model_counts=%s",
                    sum(model_counts.values()),
                    dataset.cache_size(),
                    dict(model_counts),
                )
            step_idx += 1
    finally:
        dataset.close()
        collector.shutdown()

    logger.info("Done. final_remote_ready=%s model_counts=%s", dataset.cache_size(), dict(model_counts))


def _setup_logging(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    level_name = str(data_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


if __name__ == "__main__":
    main()
