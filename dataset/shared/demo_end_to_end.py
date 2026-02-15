from __future__ import annotations

import logging
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import data_pipeline, setup_logging


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger = logging.getLogger("dataset.shared.demo_end_to_end")

    logger.info("Starting end-to-end shared dataset demo")
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    demo_cfg = cfg.get("demo", {})
    num_samples = int(demo_cfg.get("num_samples", 200))

    with data_pipeline(cfg, logger=logger, emit_run_report=False) as (dataset, collector):
        iterator = iter(dataset)
        model_counts: Counter[str] = Counter()
        layer_counts: Counter[str] = Counter()
        dataset_mix_counts: Counter[str] = Counter()
        job_mix_log: list[dict[str, int]] = []
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
                    collector.cache_size(),
                    dict(model_counts),
                )

            step_idx += 1
        logger.info("Final cache size: %s", collector.cache_size())
        logger.info("Model switching frequency: %s", dict(model_counts))
        logger.info("Top layers: %s", layer_counts.most_common(5))
        logger.info("Dataset mix distribution (from sample meta): %s", dict(dataset_mix_counts))
        if job_mix_log:
            logger.info("Dataset mix per interleaved collector job: %s", job_mix_log[:20])


if __name__ == "__main__":
    main()
