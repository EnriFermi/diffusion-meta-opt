from __future__ import annotations

import contextlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
from omegaconf import DictConfig, OmegaConf

from dataset.data_raw.providers.hf.auth import init_hf_auth
from dataset.logging_utils import configure_process_logging
from dataset.models.registry import create_model
from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset


def setup_logging(cfg: DictConfig, rank: int = 0) -> Path:
    log_path = configure_process_logging(cfg=cfg, role="train", rank=rank, force=True)
    logging.getLogger("dataset.logging").info("Run log file: %s", log_path)
    return log_path


def _read_dataset_size_from_meta(data_root: Path, dataset_name: str) -> int | None:
    meta_path = data_root / dataset_name / "meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("dataset_size")
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _safe_load_dataset_builder(repo: str, subset: str | None, token: str | None, trust_remote_code: bool):
    from datasets import load_dataset_builder

    kwargs = {"trust_remote_code": bool(trust_remote_code)}
    if token:
        try:
            return load_dataset_builder(repo, subset, token=token, **kwargs)
        except TypeError:
            try:
                return load_dataset_builder(repo, subset, use_auth_token=token, **kwargs)
            except TypeError:
                return load_dataset_builder(repo, subset, **kwargs)
    return load_dataset_builder(repo, subset, **kwargs)


def _resolve_dataset_size_via_builder(dataset_cfg: dict, token: str | None) -> int | None:
    hf_cfg = dataset_cfg.get("hf", {}) if isinstance(dataset_cfg, dict) else {}
    repo = hf_cfg.get("repo")
    if not repo:
        return None

    subset = hf_cfg.get("subset")
    split = str(hf_cfg.get("split", "train"))
    base_split = split.split("[", 1)[0]

    try:
        builder = _safe_load_dataset_builder(
            repo=str(repo),
            subset=str(subset) if subset is not None else None,
            token=token,
            trust_remote_code=bool(hf_cfg.get("trust_remote_code", False)),
        )
        splits = getattr(builder.info, "splits", None) or {}
        target = splits.get(split) or splits.get(base_split)
        if target is None:
            return None
        value = getattr(target, "num_examples", None)
        if isinstance(value, int) and value >= 0:
            return value
        return None
    except Exception:
        return None


def _inspect_model_linear_shapes(
    model_name: str,
    model_cfg: dict,
    global_cfg: DictConfig,
) -> tuple[int | None, dict[tuple[int, int], int]]:
    logger = logging.getLogger("dataset.runtime")
    runtime_cfg = dict(model_cfg)
    runtime_cfg["device"] = "cpu"
    runtime_cfg["local_files_only"] = False
    runtime_cfg["release_device_on_unload"] = True
    runtime_cfg["empty_cuda_cache_on_unload"] = False

    model = create_model(model_name, cfg=runtime_cfg, global_cfg=global_cfg)
    total_layers: int | None = None
    shape_hist: dict[tuple[int, int], int] = {}

    try:
        model.load()
        raw_model = getattr(model, "_model", None)
        if raw_model is None:
            return None, {}

        counts: Counter[tuple[int, int]] = Counter()
        total = 0
        for module in raw_model.modules():
            if isinstance(module, torch.nn.Linear):
                out_f = int(getattr(module, "out_features", 0))
                in_f = int(getattr(module, "in_features", 0))
                counts[(out_f, in_f)] += 1
                total += 1

        total_layers = total
        shape_hist = dict(counts)
    except Exception as exc:
        logger.warning("Could not inspect linear layers for model '%s': %s", model_name, exc)
    finally:
        with contextlib.suppress(Exception):
            model.unload()

    return total_layers, shape_hist


def log_run_configuration(
    cfg: DictConfig,
    collector: CollectorService,
    logger: logging.Logger | None = None,
) -> None:
    logger_local = logger or logging.getLogger("dataset.runtime")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger_local.warning("Could not resolve config for run summary")
        return

    token = init_hf_auth(cfg_dict, allow_missing_token=True)
    data_root = Path(str(cfg.data.path))

    index = collector.compat_index
    dataset_cfgs = index.dataset_cfgs
    enabled_datasets = [name for name, ds_cfg in dataset_cfgs.items() if bool(ds_cfg.get("enabled", True))]
    collectable_models = index.get_models()

    logger_local.info("========== RUN CONFIG SUMMARY ==========")
    logger_local.info("Raw datasets and assigned models:")

    dataset_sizes: dict[str, int | None] = {}
    for dataset_name in enabled_datasets:
        ds_cfg = dict(dataset_cfgs[dataset_name])
        models = [str(item) for item in ds_cfg.get("models", []) if str(item) in collectable_models]

        size = _read_dataset_size_from_meta(data_root=data_root, dataset_name=dataset_name)
        if size is None:
            size = _resolve_dataset_size_via_builder(ds_cfg, token=token)
        dataset_sizes[dataset_name] = size

        size_text = str(size) if size is not None else "unknown"
        models_text = ", ".join(models) if models else "-"

        logger_local.info("- %s:", dataset_name)
        logger_local.info("    objects number: %s", size_text)
        logger_local.info("    applying models: %s", models_text)

    logger_local.info("Initialized models and linear-layer shapes:")
    model_layer_totals: dict[str, int] = {}

    for model_name in collectable_models:
        model_cfg = index.get_model_cfg(model_name)
        in_use_count = len(index.get_datasets_for_model(model_name))

        total_layers, shape_hist = _inspect_model_linear_shapes(model_name, model_cfg, cfg)
        if total_layers is not None:
            model_layer_totals[model_name] = int(total_layers)

        logger_local.info("- %s: in use of %s different datasets", model_name, in_use_count)
        if not shape_hist:
            logger_local.info("    layers with shape ... x ...: unknown (inspection failed or no Linear layers)")
        else:
            for (out_f, in_f), count in sorted(shape_hist.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
                logger_local.info("    layers with shape %s x %s: %s", out_f, in_f, count)

    known_sizes = [value for value in dataset_sizes.values() if isinstance(value, int)]
    unknown_size_count = len(dataset_sizes) - len(known_sizes)
    objects_total_text = str(sum(known_sizes))
    if unknown_size_count > 0:
        objects_total_text += f" (+ unknown for {unknown_size_count} dataset(s))"

    linear_layers_total = sum(model_layer_totals.values())

    logger_local.info("Overall totals:")
    logger_local.info("- Datasets in total: %s", len(enabled_datasets))
    logger_local.info("- Models in total: %s", len(collectable_models))
    logger_local.info("- Objects in raw data total: %s", objects_total_text)
    logger_local.info("- Linear layers in total: %s", linear_layers_total)
    logger_local.info("=======================================")


@contextlib.contextmanager
def data_pipeline(
    cfg: DictConfig,
    *,
    start_collector: bool = True,
    predownload_models: bool | None = None,
    logger: logging.Logger | None = None,
    emit_run_report: bool = True,
    rank: int = 0,
) -> Iterator[tuple[SharedModelDataset, CollectorService]]:
    logger_local = logger or logging.getLogger("dataset.runtime")

    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    should_predownload = bool(cfg.train.get("predownload_models", False))
    if predownload_models is not None:
        should_predownload = bool(predownload_models)

    if start_collector and should_predownload:
        logger_local.info("Predownloading model artifacts before collector start")
        collector.predownload_models()

    if start_collector:
        collector.start()

    logger_local.info(
        "Data pipeline initialized: collector_mode=%s streaming_mode=%s cache_metric=%s",
        collector.collector_mode,
        collector.streaming_mode,
        dataset.cache_size(),
    )

    if emit_run_report and rank == 0 and start_collector:
        log_run_configuration(cfg, collector, logger=logger_local)

    try:
        yield dataset, collector
    finally:
        dataset.close()
        collector.shutdown()
        logger_local.info("Training entrypoint shutdown complete")


__all__ = [
    "data_pipeline",
    "log_run_configuration",
    "setup_logging",
    "data_raw",
    "models",
    "shared",
]
