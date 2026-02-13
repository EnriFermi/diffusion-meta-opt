from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from dataset.data_raw.core.image_utils import build_tensor_transform
from dataset.shared.compatibility_index import CompatibilityIndex
from dataset.shared.raw_dataset_pool import RawDatasetPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one raw virtual dataset")
    parser.add_argument("dataset_name", type=str, help="Dataset adapter name (e.g. flickr30k)")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to fetch")
    parser.add_argument(
        "--output-format",
        choices=["pil", "tensor"],
        default="pil",
        help="Batch output format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("inspect_dataset")

    conf_dir = Path(__file__).resolve().parents[3] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config")

    with open_dict(cfg):
        cfg.data.enabled_datasets = [args.dataset_name]

    logger.info("Inspecting dataset: %s", args.dataset_name)
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    index = CompatibilityIndex(cfg)
    pool = RawDatasetPool(cfg=cfg, index=index)

    try:
        pool.start()

        pil_images, source_ids, _ = pool.get_pil_batch(args.dataset_name, args.n)
        mix = Counter([args.dataset_name for _ in pil_images])

        logger.info("Returned samples: %s", len(source_ids))
        logger.info("Dataset mix: %s", dict(mix))
        logger.info("Source ids: %s", source_ids)

        if args.output_format == "tensor":
            image_size = int(cfg.data.get("image_size", 224))
            transform = build_tensor_transform(image_size=image_size)
            tensor_batch = torch.stack([transform(image) for image in pil_images], dim=0)
            logger.info("Tensor shape: %s", tuple(tensor_batch.shape))

        logger.info("Dataset stats: %s", pool.stats().get(args.dataset_name, {}))
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
