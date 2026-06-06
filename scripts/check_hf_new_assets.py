from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset.shared.compatibility_index import CompatibilityIndex
from dataset.shared.raw_dataset_pool import RawDatasetPool
from dataset.models.registry import create_model

DEFAULT_DATASETS = [
    "sun397",
    "food101",
    "rvl_cdip",
    "chexpert",
    "bigearthnet",
    "openimages_v7",
    "pascal_voc_2012",
]

DEFAULT_MODELS = [
    "vit_large_p16_224",
    "clip_vit_l14",
    "siglip_so400m_p14_384",
    "donut_rvlcdip",
    "trocr_large_printed",
    "detr_resnet50",
    "segformer_b5_cityscapes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check HF loading for selected datasets and models")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset config names from conf/data/datasets (default: new datasets)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model config names from conf/data/models (default: new models)",
    )
    parser.add_argument("--dataset-samples", type=int, default=1, help="How many samples to request per dataset")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset checks")
    parser.add_argument("--skip-models", action="store_true", help="Skip model checks")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    return parser.parse_args()


def _compose_base_cfg() -> DictConfig:
    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="big_vae/train/default")

    with open_dict(cfg):
        if "logging" in cfg:
            cfg.logging.file_name = "check_hf_new_assets.log"
            cfg.logging.file_path = f"{cfg.logging.dir}/{cfg.logging.file_name}"

    return cfg


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _check_dataset(base_cfg: DictConfig, dataset_name: str, samples: int) -> dict[str, object]:
    cfg = _clone_cfg(base_cfg)

    with open_dict(cfg):
        cfg.data.enabled_datasets = [dataset_name]
        if cfg.data.get("dataset_overrides") is None:
            cfg.data.dataset_overrides = {}
        cfg.data.dataset_overrides[dataset_name] = {
            "enabled": True,
            "cache": {
                "chunk_size_images": max(8, min(int(samples) * 2, 32)),
                "num_chunks_kept": 1,
            },
            "worker": {
                "startup_get_timeout_s": 90.0,
            },
        }

    index = CompatibilityIndex(cfg)
    pool = RawDatasetPool(cfg=cfg, index=index)

    try:
        pool.start()
        pil_images, source_ids, _ = pool.get_pil_batch(dataset_name, max(1, int(samples)))
    finally:
        pool.shutdown()

    if not pil_images:
        raise RuntimeError(f"Dataset '{dataset_name}' returned zero samples")

    # CheXpert is grayscale at source; pipeline should always emit RGB PIL.
    if dataset_name == "chexpert" and getattr(pil_images[0], "mode", None) != "RGB":
        raise RuntimeError(
            f"Dataset '{dataset_name}' returned image mode={getattr(pil_images[0], 'mode', None)}; expected RGB"
        )

    first_size = None
    if hasattr(pil_images[0], "size"):
        first_size = tuple(pil_images[0].size)

    return {
        "num_samples": len(pil_images),
        "first_source_id": source_ids[0] if source_ids else None,
        "first_image_size": first_size,
    }


def _check_model(base_cfg: DictConfig, model_name: str) -> dict[str, object]:
    model_cfg_path = Path(__file__).resolve().parents[1] / "conf" / "data" / "models" / f"{model_name}.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")

    model_cfg_obj = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=True)
    if not isinstance(model_cfg_obj, dict):
        raise TypeError(f"Invalid model config payload for {model_name}: {type(model_cfg_obj)}")

    runtime_cfg = dict(model_cfg_obj)
    runtime_cfg["device"] = "cpu"
    runtime_cfg["local_files_only"] = False
    runtime_cfg["release_device_on_unload"] = True
    runtime_cfg["empty_cuda_cache_on_unload"] = False

    model = create_model(model_name, cfg=runtime_cfg, global_cfg=base_cfg)
    dummy_image = Image.new("RGB", (256, 256), color=(127, 127, 127))

    try:
        model.load()
        records = model.run([dummy_image])
    finally:
        model.unload()

    return {
        "num_layer_records": len(records),
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("check_hf_new_assets")

    base_cfg = _compose_base_cfg()

    failures: list[str] = []

    if not args.skip_datasets:
        logger.info("Checking datasets: %s", args.datasets)
        for dataset_name in args.datasets:
            try:
                info = _check_dataset(base_cfg, dataset_name=dataset_name, samples=args.dataset_samples)
                logger.info("[OK][dataset] %s -> %s", dataset_name, info)
            except Exception as exc:
                message = f"[FAIL][dataset] {dataset_name}: {exc}"
                failures.append(message)
                logger.error(message)
                if args.fail_fast:
                    break

    if not args.skip_models and (not failures or not args.fail_fast):
        logger.info("Checking models: %s", args.models)
        for model_name in args.models:
            try:
                info = _check_model(base_cfg, model_name=model_name)
                logger.info("[OK][model] %s -> %s", model_name, info)
            except Exception as exc:
                message = f"[FAIL][model] {model_name}: {exc}"
                failures.append(message)
                logger.error(message)
                if args.fail_fast:
                    break

    if failures:
        logger.error("HF smoke-check finished with %s failure(s)", len(failures))
        for line in failures:
            logger.error("%s", line)
        raise SystemExit(1)

    logger.info("HF smoke-check completed successfully")


if __name__ == "__main__":
    main()
