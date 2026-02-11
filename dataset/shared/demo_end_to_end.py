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
    logger = logging.getLogger("dataset.shared.demo_end_to_end")

    logger.info("Starting end-to-end shared dataset demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    demo_cfg = cfg.get("demo", {})
    num_samples = int(demo_cfg.get("num_samples", 200))

    collector = CollectorService(cfg)
    collector.predownload_models()
    collector.start()

    dataset = SharedModelDataset(collector)
    iterator = iter(dataset)

    model_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    dataset_mix_counts: Counter[str] = Counter()
    job_mix_log: list[dict[str, int]] = []

    try:
        step_idx = 0
        while sum(model_counts.values()) < num_samples:
            if not collector.is_async_mode:
                collected_stats = dataset.maybe_collect(step_idx)
                for item in collected_stats:
                    job_mix_log.append(dict(item.dataset_mix))

            sample = next(iterator)
            model_counts[sample.model_name] += 1
            layer_counts[sample.layer_name] += 1

            image_meta = sample.meta.get("image_meta", [])
            for item in image_meta:
                dataset_name = item.get("dataset_name")
                if dataset_name:
                    dataset_mix_counts[str(dataset_name)] += 1

            if sum(model_counts.values()) % 20 == 0:
                logger.info(
                    "consumed=%s cache=%s model_counts=%s",
                    sum(model_counts.values()),
                    collector.cache.size(),
                    dict(model_counts),
                )

            step_idx += 1
    finally:
        collector.shutdown()

    logger.info("Final cache size: %s", collector.cache.size())
    logger.info("Model switching frequency: %s", dict(model_counts))
    logger.info("Top layers: %s", layer_counts.most_common(5))
    logger.info("Dataset mix distribution (from sample meta): %s", dict(dataset_mix_counts))
    if job_mix_log:
        logger.info("Dataset mix per interleaved collector job: %s", job_mix_log[:20])


def _setup_logging(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    level_name = str(data_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


if __name__ == "__main__":
    main()
