from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import asdict
from queue import Empty, Full
from typing import Any

from omegaconf import DictConfig

from dataset.data_raw.core.config import to_plain_dict
from dataset.logging_utils import configure_root_logging
from dataset.models.model_pool import ModelPool
from dataset.shared.atomizer import atomize
from dataset.shared.cache import SharedSampleCache
from dataset.shared.compatibility_index import (
    CompatibilityIndex,
    normalize_device,
    resolve_collector_mode,
    resolve_train_device,
)
from .load_report import LoadReportWriter
from dataset.shared.model_scheduler import ModelScheduler
from dataset.shared.raw_dataset_pool import RawDatasetPool
from dataset.shared.streaming.factory import (
    build_chunk_reader,
    build_chunk_store,
    build_chunk_writer,
    resolve_streaming_cfg,
)
from dataset.shared.types import CollectorJobStats, SharedSample


class _SampleSink(ABC):
    @abstractmethod
    def emit(self, sample: SharedSample) -> None:
        raise NotImplementedError

    @abstractmethod
    def needs_fill(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_low(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def size_metric(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        raise NotImplementedError


class _InMemorySampleSink(_SampleSink):
    def __init__(self, cache: SharedSampleCache) -> None:
        self.cache = cache

    def emit(self, sample: SharedSample) -> None:
        self.cache.put(sample)

    def needs_fill(self) -> bool:
        return self.cache.needs_fill()

    def is_low(self) -> bool:
        return self.cache.is_low()

    def size_metric(self) -> int:
        return self.cache.size()

    def close(self) -> None:
        self.cache.close()

    def stats(self) -> dict[str, Any]:
        return {
            "sink": "in_memory",
            "size": self.cache.size(),
            "capacity_samples": self.cache.max_items,
            "fill_target_samples": self.cache.fill_target,
            "low_watermark_samples": self.cache.low_watermark,
        }


class _ChunkedSampleSink(_SampleSink):
    def __init__(
        self,
        mode: str,
        store: Any,
        writer: Any,
        refill_after_consumed_chunks: int,
        stripe_window_chunks: int | None,
    ) -> None:
        self.mode = mode
        self.store = store
        self.writer = writer
        self.refill_after_consumed_chunks = max(1, int(refill_after_consumed_chunks))
        self.stripe_window_chunks = (
            max(1, int(stripe_window_chunks))
            if stripe_window_chunks is not None
            else self.refill_after_consumed_chunks
        )

        self._fill_active = False
        self._window_id = 0

    def emit(self, sample: SharedSample) -> None:
        self.writer.append(sample)

    def needs_fill(self) -> bool:
        state = self.store.capacity_state()
        can_accept = bool(state.get("can_accept", True))
        ready, upper_bound = self._resolve_ready_and_upper_bound(state)
        start_threshold = max(0, upper_bound - self.refill_after_consumed_chunks)

        if self._fill_active:
            if not can_accept or ready >= upper_bound:
                self._fill_active = False
                self.writer.close_window(flush_partial=False)
                return False
            return True

        should_start = can_accept and ready <= start_threshold
        if not should_start:
            return False

        self.writer.open_window(window_chunks=self.stripe_window_chunks, window_id=self._window_id)
        self._window_id += 1
        self._fill_active = True
        return True

    def is_low(self) -> bool:
        state = self.store.capacity_state()
        ready, upper_bound = self._resolve_ready_and_upper_bound(state)
        start_threshold = max(0, upper_bound - self.refill_after_consumed_chunks)
        return ready <= start_threshold

    def size_metric(self) -> int:
        state = self.store.capacity_state()
        if "ready_chunks" in state:
            return int(state["ready_chunks"])
        return self.store.count_ready()

    def close(self) -> None:
        if self._fill_active:
            self.writer.close_window(flush_partial=True)
            self._fill_active = False
        self.writer.close()

    def stats(self) -> dict[str, Any]:
        state = self.store.capacity_state()
        payload = {
            "sink": "chunked",
            "mode": self.mode,
            "size_metric": self.size_metric(),
            "backend": state.get("backend"),
            "fill_active": self._fill_active,
            "refill_after_consumed_chunks": self.refill_after_consumed_chunks,
            "stripe_window_chunks": self.stripe_window_chunks,
            "window_id": self._window_id,
        }
        payload.update({f"backend_{k}": v for k, v in state.items()})
        payload.update({f"writer_{k}": v for k, v in self.writer.stats().items()})
        return payload

    def _resolve_ready_and_upper_bound(self, state: dict[str, Any]) -> tuple[int, int]:
        ready = int(state.get("ready_chunks", self.store.count_ready()))

        if self.mode == "local_disk":
            upper = max(1, int(state.get("max_ready_chunks", 1)))
            return ready, upper

        if self.mode == "s3_bridge":
            upper = max(1, int(state.get("max_remote_chunks", 1)))
            return ready, upper

        return ready, max(1, ready)


class CollectorService:
    """Model-first collector orchestration with async and interleaved modes."""

    def __init__(
        self,
        cfg: DictConfig | dict[str, Any],
        cache: SharedSampleCache | None = None,
        status_queue: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.cfg_dict = to_plain_dict(cfg)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.compat_index = CompatibilityIndex(self.cfg)
        self.collector_mode = resolve_collector_mode(self.cfg_dict)

        collector_cfg = self.cfg_dict.get("collector", {})
        in_memory_cfg = collector_cfg.get("in_memory_buffer", {})

        self.streaming_cfg = resolve_streaming_cfg(self.cfg_dict.get("streaming", {}))
        consumer_cfg = dict(self.streaming_cfg.get("consumer", {}))
        if consumer_cfg.get("random_seed") is None:
            data_cfg = self.cfg_dict.get("data") or {}
            if data_cfg.get("seed") is not None:
                consumer_cfg["random_seed"] = int(data_cfg.get("seed"))
        self.streaming_cfg["consumer"] = consumer_cfg
        self.streaming_mode = str(self.streaming_cfg.get("mode", "none"))
        self.streaming_enabled = self.streaming_mode != "none"

        self.cache: SharedSampleCache | None = None
        if not self.streaming_enabled:
            self.cache = cache or SharedSampleCache(
                max_items=int(in_memory_cfg.get("capacity_samples", 5000)),
                fill_target=int(in_memory_cfg.get("fill_target_samples", 5000)),
                low_watermark=int(in_memory_cfg.get("low_watermark_samples", 3000)),
            )

        self.collector_device = normalize_device(collector_cfg.get("device"))
        self.train_device = resolve_train_device(self.cfg_dict)

        if self.collector_mode == "async":
            if self.collector_device is None:
                raise ValueError("collector.device must be set for async collector mode")
            if self.train_device is not None and self.collector_device == self.train_device:
                raise ValueError("async collector mode requires collector.device != train.device")

        self.model_selection_strategy = str(collector_cfg.get("model_selection_strategy", "round_robin_shuffled"))
        self.jobs_per_selected_model = int(collector_cfg.get("jobs_per_selected_model", 2))
        self.dataset_sampling_strategy = str(collector_cfg.get("dataset_sampling_strategy", "weighted_random"))
        self.max_dataset_fraction_per_batch = float(collector_cfg.get("max_dataset_fraction_per_batch", 0.6))
        self.num_inflight_jobs = max(1, int(collector_cfg.get("num_inflight_jobs", 2)))
        self.status_emit_interval_s = max(0.1, float(collector_cfg.get("status_emit_interval_s", 2.0)))
        self.status_queue_max_items = max(16, int(collector_cfg.get("status_queue_max_items", 2048)))

        self.interleaved_cfg = collector_cfg.get("interleaved_schedule", {})
        self.interleaved_every_n_steps = max(
            1,
            int(self.interleaved_cfg.get("collect_every_n_train_steps", 30)),
        )
        self.interleaved_burst_jobs = max(
            1,
            int(self.interleaved_cfg.get("collector_jobs_per_cycle", 1)),
        )

        self.atom_cfg = collector_cfg.get("layer_output_splitting", {})
        self._validate_layer_output_splitting(self.atom_cfg)

        max_loaded = int(collector_cfg.get("max_loaded_models", 1))
        if max_loaded != 1:
            raise ValueError("collector.max_loaded_models must be 1 (single collector model on collector GPU)")

        self._runtime_ready = False
        self._raw_pool: RawDatasetPool | None = None
        self._model_pool: ModelPool | None = None
        self._scheduler: ModelScheduler | None = None
        self._sink: _SampleSink | None = None
        self._load_report = LoadReportWriter(self.cfg_dict)

        self._model_run_id = 0
        self._jobs_total = 0
        self._jobs_by_model: Counter[str] = Counter()
        self._items_emitted = 0

        self._ctx = mp.get_context("spawn")
        self._stop_event: Any | None = None
        self._process: mp.Process | None = None
        self._status_queue: Any | None = status_queue
        self._last_status_emit_ts = 0.0
        self._async_last_status: dict[str, Any] | None = None
        self._async_recent_jobs: deque[dict[str, Any]] = deque(maxlen=128)
        self._async_events_dropped = 0

    @property
    def is_async_mode(self) -> bool:
        return self.collector_mode == "async"

    def is_async_process_alive(self) -> bool:
        if not self.is_async_mode:
            return True
        return bool(self._process is not None and self._process.is_alive())

    def assert_healthy(self) -> None:
        """
        Fail fast when async collector process has exited.

        Without this guard, consumers may block forever waiting for new samples
        after collector crash.
        """
        if not self.is_async_mode:
            return

        self._drain_status_queue()
        if self._process is None:
            return
        if self._process.is_alive():
            return

        exit_code = self._process.exitcode
        raise RuntimeError(f"Async collector process exited unexpectedly (exitcode={exit_code})")

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
            release_device_on_unload=True,
            empty_cuda_cache_on_unload=True,
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

        if self._status_queue is not None:
            try:
                self._status_queue.close()
            except Exception:
                pass
            self._status_queue = None

        self._shutdown_runtime_components()

    def run_forever(self, stop_event: Any | None = None) -> None:
        self._ensure_runtime_ready()
        assert self._sink is not None

        active_stop_event = stop_event
        if active_stop_event is None:
            active_stop_event = self._ctx.Event()

        while not active_stop_event.is_set():
            if self._sink.needs_fill():
                for _ in range(self.num_inflight_jobs):
                    if active_stop_event.is_set():
                        break
                    if not self._sink.needs_fill():
                        break
                    self.collect_one_job()
            else:
                time.sleep(0.1)

            self._maybe_emit_status_event(event_type="heartbeat")

    def collect_burst(self, num_jobs: int) -> list[CollectorJobStats]:
        self._ensure_runtime_ready()
        assert self._sink is not None

        stats: list[CollectorJobStats] = []
        for _ in range(max(1, int(num_jobs))):
            if not self._sink.needs_fill():
                break
            stats.append(self.collect_one_job())
        return stats

    def maybe_collect(self, step_idx: int) -> list[CollectorJobStats]:
        if self.is_async_mode:
            return []

        self._ensure_runtime_ready()
        assert self._sink is not None

        if not self._sink.is_low():
            return []

        if step_idx % self.interleaved_every_n_steps != 0:
            return []

        return self.collect_burst(self.interleaved_burst_jobs)

    def collect_one_job(self) -> CollectorJobStats:
        self._ensure_runtime_ready()

        assert self._scheduler is not None
        assert self._raw_pool is not None
        assert self._model_pool is not None
        assert self._sink is not None

        if not self._sink.needs_fill():
            return CollectorJobStats(
                model_name="none",
                num_images=0,
                num_layers=0,
                num_samples_emitted=0,
                dataset_mix={},
                duration_s=0.0,
            )

        started = time.time()

        model_name = self._scheduler.next_model()
        model_cfg = self.compat_index.get_model_cfg(model_name)
        batch_size = max(1, int(model_cfg.get("batch_size", 1)))

        pil_batch, image_meta = self._raw_pool.sample_mixed_batch(
            model_name=model_name,
            batch_size=batch_size,
            dataset_sampling_strategy=self.dataset_sampling_strategy,
            max_dataset_fraction_per_batch=self.max_dataset_fraction_per_batch,
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
                self._sink.emit(shared_sample)
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
            "collector job model=%s images=%s layers=%s emitted=%s size=%s mode=%s",
            model_name,
            stats.num_images,
            stats.num_layers,
            emitted,
            self.cache_size(),
            self.streaming_mode,
        )
        self._emit_status_event(event_type="job", job_stats=stats)

        return stats

    def cache_size(self) -> int:
        if self.is_async_mode and not self._runtime_ready:
            self._drain_status_queue()
        if self.is_async_mode and self._sink is None and self._async_last_status is not None:
            cache_size_value = self._async_last_status.get("cache_size")
            if cache_size_value is not None:
                return int(cache_size_value)

        if self.streaming_mode == "none":
            if self.cache is None:
                return 0
            return self.cache.size()

        if self._sink is not None:
            return self._sink.size_metric()

        store = build_chunk_store(self.streaming_cfg)
        if store is None:
            return 0
        return store.count_ready()

    def create_chunk_reader(self):
        if self.streaming_mode == "none":
            return None
        store = build_chunk_store(self.streaming_cfg)
        if store is None:
            return None
        return build_chunk_reader(self.streaming_cfg, store)

    def stats(self) -> dict[str, Any]:
        if self.is_async_mode and not self._runtime_ready:
            self._drain_status_queue()

        payload = {
            "mode": self.collector_mode,
            "streaming_mode": self.streaming_mode,
            "collector_device": self.collector_device,
            "train_device": self.train_device,
            "cache_size": self.cache_size(),
            "jobs_total": self._jobs_total,
            "jobs_by_model": dict(self._jobs_by_model),
            "items_emitted": self._items_emitted,
            "scheduler": self._scheduler.stats() if self._scheduler else None,
            "sink": self._sink.stats() if self._sink else None,
            "status_emit_interval_s": self.status_emit_interval_s,
            "status_queue_max_items": self.status_queue_max_items,
        }

        if self.is_async_mode:
            payload["async_process_alive"] = bool(self._process is not None and self._process.is_alive())
            payload["async_events_dropped"] = int(self._async_events_dropped)
            payload["async_recent_jobs"] = list(self._async_recent_jobs)
            payload["async_last_status"] = self._async_last_status

            if self._async_last_status is not None:
                payload["cache_size"] = int(self._async_last_status.get("cache_size", payload["cache_size"]))
                payload["jobs_total"] = int(self._async_last_status.get("jobs_total", payload["jobs_total"]))
                payload["items_emitted"] = int(self._async_last_status.get("items_emitted", payload["items_emitted"]))
                queue_jobs = self._async_last_status.get("jobs_by_model")
                if isinstance(queue_jobs, dict):
                    payload["jobs_by_model"] = {
                        str(name): int(value)
                        for name, value in queue_jobs.items()
                    }
                if payload["scheduler"] is None:
                    payload["scheduler"] = self._async_last_status.get("scheduler")
                if payload["sink"] is None:
                    payload["sink"] = self._async_last_status.get("sink")

        return payload

    def _ensure_runtime_ready(self) -> None:
        if self._runtime_ready:
            return

        model_cfgs = self.compat_index.get_model_cfgs()
        collectable_models = self.compat_index.get_models()
        if not collectable_models:
            raise ValueError("No collectable models found in CompatibilityIndex")

        expected_datasets = sorted(str(name) for name in self.compat_index.dataset_cfgs.keys())
        self._load_report.set_expected(datasets=expected_datasets, models=collectable_models)
        self._load_report.mark_runtime_event(
            "runtime_init_start",
            {
                "collector_mode": self.collector_mode,
                "streaming_mode": self.streaming_mode,
                "collector_device": self.collector_device,
                "train_device": self.train_device,
            },
        )
        self.logger.info(
            "Load report paths: summary=%s events=%s",
            self._load_report.summary_path,
            self._load_report.events_path,
        )

        try:
            self._raw_pool = RawDatasetPool(cfg=self.cfg, index=self.compat_index, load_report=self._load_report)
            self._raw_pool.start()

            collector_cfg = self.cfg_dict.get("collector", {})
            max_loaded = int(collector_cfg.get("max_loaded_models", 1))
            pin_gpu = bool(collector_cfg.get("pin_gpu", self.is_async_mode))

            release_cfg = collector_cfg.get("release_device_on_unload")
            if release_cfg is None:
                release_device_on_unload = not pin_gpu
            else:
                release_device_on_unload = bool(release_cfg)

            empty_cfg = collector_cfg.get("empty_cuda_cache_on_unload")
            if empty_cfg is None:
                empty_cuda_cache_on_unload = not pin_gpu
            else:
                empty_cuda_cache_on_unload = bool(empty_cfg)

            runtime_device = self.collector_device
            if runtime_device is None:
                runtime_device = self.train_device

            self._model_pool = ModelPool(
                global_cfg=self.cfg,
                model_cfgs=model_cfgs,
                device_override=runtime_device,
                max_loaded_models=max_loaded,
                runtime_local_only=bool(collector_cfg.get("runtime_local_only", False)),
                release_device_on_unload=release_device_on_unload,
                empty_cuda_cache_on_unload=empty_cuda_cache_on_unload,
                load_report=self._load_report,
            )

            data_cfg = self.cfg_dict.get("data") or {}
            seed = int(data_cfg.get("seed", 0))

            self._scheduler = ModelScheduler(
                model_names=collectable_models,
                model_weights=self.compat_index.get_model_weights(),
                policy=self.model_selection_strategy,
                burst_jobs=self.jobs_per_selected_model,
                seed=seed,
            )

            self._sink = self._build_sink()
            self._runtime_ready = True
            self._load_report.mark_runtime_event("runtime_init_ready")
        except Exception as exc:
            self._load_report.mark_runtime_event(
                "runtime_init_failed",
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    def _build_sink(self) -> _SampleSink:
        if self.streaming_mode == "none":
            if self.cache is None:
                raise ValueError("In-memory collector mode requires SharedSampleCache")
            return _InMemorySampleSink(self.cache)

        store = build_chunk_store(self.streaming_cfg)
        if store is None:
            raise ValueError(f"streaming.mode={self.streaming_mode} requires a chunk store")

        writer = build_chunk_writer(self.streaming_cfg, store=store)
        producer_cfg = dict(self.streaming_cfg.get("producer", {}))
        refill_after = int(producer_cfg.get("refill_after_consumed_chunks", 1))
        stripe_window = producer_cfg.get("stripe_window_chunks")
        return _ChunkedSampleSink(
            mode=self.streaming_mode,
            store=store,
            writer=writer,
            refill_after_consumed_chunks=refill_after,
            stripe_window_chunks=int(stripe_window) if stripe_window is not None else None,
        )

    def _shutdown_runtime_components(self) -> None:
        if self._sink is not None:
            try:
                self._sink.close()
            except Exception as exc:
                self.logger.warning("Failed to close collector sink: %s", exc)
            self._sink = None

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
        cache_arg = self.cache if self.streaming_mode == "none" else None
        if self._status_queue is None:
            self._status_queue = self._ctx.Queue(maxsize=self.status_queue_max_items)

        self._process = self._ctx.Process(
            target=collector_process_main,
            args=(cfg_dict, cache_arg, self._stop_event, self._status_queue),
            daemon=False,  # must be False: collector spawns dataset workers (daemon cannot have children)
            name="collector_service",
        )
        self._process.start()
        self.logger.info("Started async collector process pid=%s", self._process.pid)

    def _next_run_id(self) -> int:
        self._model_run_id += 1
        return self._model_run_id

    def _maybe_emit_status_event(self, event_type: str) -> None:
        now = time.time()
        if now - self._last_status_emit_ts < self.status_emit_interval_s:
            return
        self._emit_status_event(event_type=event_type, job_stats=None)

    def _emit_status_event(self, event_type: str, job_stats: CollectorJobStats | None) -> None:
        if self._status_queue is None:
            return

        payload: dict[str, Any] = {
            "type": str(event_type),
            "timestamp": float(time.time()),
            "mode": self.collector_mode,
            "streaming_mode": self.streaming_mode,
            "cache_size": int(self.cache_size()),
            "jobs_total": int(self._jobs_total),
            "jobs_by_model": {name: int(value) for name, value in self._jobs_by_model.items()},
            "items_emitted": int(self._items_emitted),
            "scheduler": self._scheduler.stats() if self._scheduler else None,
            "sink": self._sink.stats() if self._sink else None,
            "events_dropped": int(self._async_events_dropped),
        }
        if job_stats is not None:
            payload["job_stats"] = asdict(job_stats)

        try:
            self._status_queue.put_nowait(payload)
            self._last_status_emit_ts = payload["timestamp"]
        except Full:
            self._async_events_dropped += 1
        except Exception:
            pass

    def _drain_status_queue(self) -> None:
        if self._status_queue is None:
            return

        while True:
            try:
                payload = self._status_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

            if not isinstance(payload, dict):
                continue

            self._async_last_status = payload
            jobs_total = payload.get("jobs_total")
            if jobs_total is not None:
                try:
                    self._jobs_total = max(self._jobs_total, int(jobs_total))
                except Exception:
                    pass

            items_emitted = payload.get("items_emitted")
            if items_emitted is not None:
                try:
                    self._items_emitted = max(self._items_emitted, int(items_emitted))
                except Exception:
                    pass

            jobs_by_model = payload.get("jobs_by_model")
            if isinstance(jobs_by_model, dict):
                counter: Counter[str] = Counter()
                for name, value in jobs_by_model.items():
                    try:
                        counter[str(name)] = int(value)
                    except Exception:
                        continue
                if counter:
                    self._jobs_by_model = counter

            job_stats = payload.get("job_stats")
            if isinstance(job_stats, dict):
                sanitized = {
                    "timestamp": payload.get("timestamp"),
                    "model_name": str(job_stats.get("model_name", "")),
                    "num_images": int(job_stats.get("num_images", 0)),
                    "num_layers": int(job_stats.get("num_layers", 0)),
                    "num_samples_emitted": int(job_stats.get("num_samples_emitted", 0)),
                    "dataset_mix": job_stats.get("dataset_mix", {}),
                    "duration_s": float(job_stats.get("duration_s", 0.0)),
                }
                self._async_recent_jobs.append(sanitized)

            dropped = payload.get("events_dropped")
            if dropped is not None:
                try:
                    self._async_events_dropped = max(self._async_events_dropped, int(dropped))
                except Exception:
                    pass

    @staticmethod
    def _validate_layer_output_splitting(atom_cfg: dict[str, Any]) -> None:
        value = atom_cfg.get("xy_samples_random_slice")
        if value is None:
            raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer")
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer")


def collector_process_main(
    cfg_dict: dict[str, Any],
    cache: SharedSampleCache | None,
    stop_event: Any,
    status_queue: Any | None = None,
) -> None:
    log_path = configure_root_logging(cfg=cfg_dict, rank=0, force=True)
    logger = logging.getLogger("collector_process")
    logger.info("Run log file: %s", log_path)

    service = CollectorService(cfg=cfg_dict, cache=cache, status_queue=status_queue)
    try:
        service.run_forever(stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("Collector process interrupted")
    finally:
        service.shutdown()


def stats_to_dict_list(items: list[CollectorJobStats]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
