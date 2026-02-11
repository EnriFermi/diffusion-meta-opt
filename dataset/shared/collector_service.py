from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import asdict
from typing import Any

from omegaconf import DictConfig

from dataset.data_raw.core.config import to_plain_dict
from dataset.models.model_pool import ModelPool
from dataset.shared.atomizer import atomize
from dataset.shared.cache import SharedSampleCache
from dataset.shared.compatibility_index import CompatibilityIndex, normalize_device, resolve_collector_mode
from dataset.shared.model_scheduler import ModelScheduler
from dataset.shared.raw_dataset_pool import RawDatasetPool
from dataset.shared.types import CollectorJobStats, SharedSample


class CollectorService:
    """Model-first collector orchestration with async and interleaved modes."""

    def __init__(self, cfg: DictConfig | dict[str, Any], cache: SharedSampleCache | None = None) -> None:
        self.cfg = cfg
        self.cfg_dict = to_plain_dict(cfg)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.compat_index = CompatibilityIndex(self.cfg)
        self.collector_mode = resolve_collector_mode(self.cfg_dict)

        collector_cfg = self.cfg_dict.get("collector", {})
        cache_cfg = collector_cfg.get("cache", {})

        self.cache = cache or SharedSampleCache(
            max_items=int(cache_cfg.get("max_items", 5000)),
            fill_target=int(cache_cfg.get("fill_target", 5000)),
            low_watermark=int(cache_cfg.get("low_watermark", 3000)),
        )

        self.collector_device = normalize_device(collector_cfg.get("device"))
        self.train_device = normalize_device(self.cfg_dict.get("train", {}).get("device"))

        if self.collector_mode == "async":
            if self.collector_device is None:
                raise ValueError("collector.device must be set for async collector mode")
            if self.train_device is not None and self.collector_device == self.train_device:
                raise ValueError("async collector mode requires collector.device != train.device")

        self.model_policy = str(collector_cfg.get("model_policy", "shuffled_cycle"))
        self.model_burst_jobs = int(collector_cfg.get("model_burst_jobs", 2))
        self.dataset_mix_policy = str(collector_cfg.get("dataset_mix_policy", "multinomial"))
        self.mix_cap_per_dataset = float(collector_cfg.get("mix_cap_per_dataset", 0.6))
        self.num_inflight_jobs = max(1, int(collector_cfg.get("num_inflight_jobs", 2)))

        self.interleaved_cfg = collector_cfg.get("interleaved", {})
        self.interleaved_every_n_steps = max(1, int(self.interleaved_cfg.get("every_n_steps", 30)))
        self.interleaved_burst_jobs = max(1, int(self.interleaved_cfg.get("burst_jobs", 1)))

        self.atom_cfg = collector_cfg.get("atomization", {})

        max_loaded = int(collector_cfg.get("max_loaded_models", 1))
        if max_loaded != 1:
            raise ValueError("collector.max_loaded_models must be 1 (single collector model on collector GPU)")

        self._runtime_ready = False
        self._raw_pool: RawDatasetPool | None = None
        self._model_pool: ModelPool | None = None
        self._scheduler: ModelScheduler | None = None

        self._model_run_id = 0
        self._jobs_total = 0
        self._jobs_by_model: Counter[str] = Counter()
        self._items_emitted = 0

        self._ctx = mp.get_context("spawn")
        self._stop_event: Any | None = None
        self._process: mp.Process | None = None

    @property
    def is_async_mode(self) -> bool:
        return self.collector_mode == "async"

    def predownload_models(self) -> None:
        model_cfgs = self.compat_index.get_model_cfgs()
        models = self.compat_index.get_models()
        if not models:
            raise ValueError("No collectable models to predownload")

        pool = ModelPool(
            global_cfg=self.cfg,
            model_cfgs=model_cfgs,
            device_override="cpu",
            max_loaded_models=1,
            runtime_local_only=False,
        )
        try:
            pool.predownload_models(models)
        finally:
            pool.unload_all()

    def start(self) -> None:
        if self.is_async_mode:
            self._start_async_process()
            return

        self._ensure_runtime_ready()

    def shutdown(self) -> None:
        if self._process is not None:
            if self._stop_event is not None:
                self._stop_event.set()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=3)
            self._process = None

        self._shutdown_runtime_components()

    def run_forever(self, stop_event: Any | None = None) -> None:
        self._ensure_runtime_ready()

        active_stop_event = stop_event
        if active_stop_event is None:
            active_stop_event = self._ctx.Event()

        while not active_stop_event.is_set():
            if self.cache.needs_fill():
                for _ in range(self.num_inflight_jobs):
                    if active_stop_event.is_set():
                        break
                    if not self.cache.needs_fill():
                        break
                    self.collect_one_job()
            else:
                time.sleep(0.1)

    def collect_burst(self, num_jobs: int) -> list[CollectorJobStats]:
        self._ensure_runtime_ready()

        stats: list[CollectorJobStats] = []
        for _ in range(max(1, int(num_jobs))):
            if not self.cache.needs_fill():
                break
            stats.append(self.collect_one_job())
        return stats

    def maybe_collect(self, step_idx: int) -> list[CollectorJobStats]:
        if self.is_async_mode:
            return []

        if not self.cache.is_low():
            return []

        if step_idx % self.interleaved_every_n_steps != 0:
            return []

        return self.collect_burst(self.interleaved_burst_jobs)

    def collect_one_job(self) -> CollectorJobStats:
        self._ensure_runtime_ready()

        assert self._scheduler is not None
        assert self._raw_pool is not None
        assert self._model_pool is not None

        started = time.time()

        model_name = self._scheduler.next_model()
        model_cfg = self.compat_index.get_model_cfg(model_name)
        batch_size = max(1, int(model_cfg.get("batch_size", 1)))

        pil_batch, image_meta = self._raw_pool.sample_mixed_batch(
            model_name=model_name,
            batch_size=batch_size,
            mixing_policy=self.dataset_mix_policy,
            mix_cap_per_dataset=self.mix_cap_per_dataset,
        )

        if not pil_batch:
            duration_s = time.time() - started
            return CollectorJobStats(
                model_name=model_name,
                num_images=0,
                num_layers=0,
                num_samples_emitted=0,
                dataset_mix={},
                duration_s=duration_s,
            )

        layer_records = self._model_pool.run(model_name=model_name, pil_batch=pil_batch)

        emitted = 0
        run_id = self._next_run_id()
        for record in layer_records:
            for shared_sample in atomize(
                layer_record=record,
                atom_cfg=self.atom_cfg,
                image_meta_list=image_meta,
                model_run_id=run_id,
            ):
                self.cache.put(shared_sample)
                emitted += 1

        mix_counter = Counter(item.dataset_name for item in image_meta)

        self._jobs_total += 1
        self._jobs_by_model[model_name] += 1
        self._items_emitted += emitted

        duration_s = time.time() - started
        stats = CollectorJobStats(
            model_name=model_name,
            num_images=len(pil_batch),
            num_layers=len(layer_records),
            num_samples_emitted=emitted,
            dataset_mix=dict(mix_counter),
            duration_s=duration_s,
        )

        self.logger.info(
            "collector job model=%s images=%s layers=%s emitted=%s cache=%s",
            model_name,
            stats.num_images,
            stats.num_layers,
            emitted,
            self.cache.size(),
        )

        return stats

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.collector_mode,
            "collector_device": self.collector_device,
            "train_device": self.train_device,
            "cache_size": self.cache.size(),
            "jobs_total": self._jobs_total,
            "jobs_by_model": dict(self._jobs_by_model),
            "items_emitted": self._items_emitted,
            "scheduler": self._scheduler.stats() if self._scheduler else None,
        }

    def _ensure_runtime_ready(self) -> None:
        if self._runtime_ready:
            return

        model_cfgs = self.compat_index.get_model_cfgs()
        collectable_models = self.compat_index.get_models()
        if not collectable_models:
            raise ValueError("No collectable models found in CompatibilityIndex")

        self._raw_pool = RawDatasetPool(cfg=self.cfg, index=self.compat_index)
        self._raw_pool.start()

        collector_cfg = self.cfg_dict.get("collector", {})
        max_loaded = int(collector_cfg.get("max_loaded_models", 1))

        runtime_device = self.collector_device
        if runtime_device is None:
            runtime_device = self.train_device

        self._model_pool = ModelPool(
            global_cfg=self.cfg,
            model_cfgs=model_cfgs,
            device_override=runtime_device,
            max_loaded_models=max_loaded,
            runtime_local_only=True,
        )

        data_cfg = self.cfg_dict.get("data") or {}
        seed = int(data_cfg.get("seed", 0))

        self._scheduler = ModelScheduler(
            model_names=collectable_models,
            model_weights=self.compat_index.get_model_weights(),
            policy=self.model_policy,
            burst_jobs=self.model_burst_jobs,
            seed=seed,
        )

        self._runtime_ready = True

    def _shutdown_runtime_components(self) -> None:
        if self._model_pool is not None:
            self._model_pool.unload_all()
            self._model_pool = None

        if self._raw_pool is not None:
            self._raw_pool.shutdown()
            self._raw_pool = None

        self._scheduler = None
        self._runtime_ready = False

    def _start_async_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._stop_event = self._ctx.Event()
        cfg_dict = to_plain_dict(self.cfg)

        self._process = self._ctx.Process(
            target=collector_process_main,
            args=(cfg_dict, self.cache, self._stop_event),
            daemon=True,
            name="collector_service",
        )
        self._process.start()
        self.logger.info("Started async collector process pid=%s", self._process.pid)

    def _next_run_id(self) -> int:
        self._model_run_id += 1
        return self._model_run_id


def collector_process_main(cfg_dict: dict[str, Any], cache: SharedSampleCache, stop_event: Any) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("collector_process")

    service = CollectorService(cfg=cfg_dict, cache=cache)
    try:
        service.run_forever(stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("Collector process interrupted")
    finally:
        service.shutdown()


def stats_to_dict_list(items: list[CollectorJobStats]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
