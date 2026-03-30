from __future__ import annotations

import faulthandler
import logging
import multiprocessing as mp
import os
import random
import signal
import time
import traceback
from collections import deque
from collections import Counter
from pathlib import Path
from queue import Empty, Full
from typing import Any

from dataset.data_raw.core.base_virtual_dataset import BaseVirtualDataset
from dataset.data_raw.core.chunk_cache import ChunkCache
from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.core.fs_utils import ensure_dir, read_json, utc_now_iso, write_json_atomic
from dataset.data_raw.core.image_utils import decode_to_pil, load_image
from dataset.data_raw.core.types import ImageSample
from dataset.data_raw.providers.hf.auth import init_hf_auth
from dataset.data_raw.providers.hf.datasets_server_sampler import DatasetServerSampler
from dataset.data_raw.providers.hf.hf_loader import load_hf_dataset
from dataset.data_raw.providers.hf.url_fetch import fetch_image_to_cache, fetch_image_to_memory
from dataset.logging_utils import configure_root_logging
from dataset.shared.shm_transport import cleanup_shared_image_rows, restore_image_rows, share_image_rows
from training.forensics import emit_fatal_report, maybe_enable_core_dumps, maybe_redirect_stdio

_FAULT_HANDLER_FILES: list[Any] = []


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:
        return f"SIG{signum}"


def _format_process_exit(exit_code: int | None) -> str:
    if exit_code is None:
        return "exitcode=None"
    if exit_code >= 0:
        return f"exitcode={exit_code}"

    signum = -int(exit_code)
    reason = f"exitcode={exit_code} ({_signal_name(signum)}/{signum})"
    if signum == getattr(signal, "SIGSEGV", -999):
        reason += " [segmentation fault]"
    elif signum == getattr(signal, "SIGABRT", -999):
        reason += " [abort]"
    elif signum == getattr(signal, "SIGBUS", -999):
        reason += " [bus error]"
    elif signum == getattr(signal, "SIGILL", -999):
        reason += " [illegal instruction]"
    elif signum == getattr(signal, "SIGFPE", -999):
        reason += " [floating point exception]"
    return reason


def _enable_fault_handler_log(log_path: Path, *, label: str, logger: logging.Logger) -> Path | None:
    fault_path = log_path.with_name(f"{log_path.stem}_fault_{label}_pid{os.getpid()}.log")
    try:
        handle = fault_path.open("a", encoding="utf-8", buffering=1)
        handle.write(f"\n=== fault-handler start pid={os.getpid()} ts={time.time():.3f} ===\n")
        faulthandler.enable(file=handle, all_threads=True)
        for sig_name in ("SIGUSR1", "SIGUSR2"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                faulthandler.register(sig, file=handle, all_threads=True, chain=True)
            except Exception:
                continue
        _FAULT_HANDLER_FILES.append(handle)
        return fault_path
    except Exception as exc:
        logger.warning("Failed to initialize worker fault-handler log file: %s", exc)
        return None


def _build_forensics_cfg(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_forensics = dataset_cfg.get("runtime_forensics", {})
    if not isinstance(runtime_forensics, dict):
        runtime_forensics = {}
    training_artifacts = runtime_forensics.get("training_artifacts", {})
    if not isinstance(training_artifacts, dict):
        training_artifacts = {}
    train_forensics = runtime_forensics.get("train_forensics", {})
    if not isinstance(train_forensics, dict):
        train_forensics = {}
    return {
        "training_artifacts": training_artifacts,
        "train": {
            "forensics": train_forensics,
        },
    }


class HFVirtualDataset(BaseVirtualDataset):
    """HF-backed virtual dataset with one prefetch process and memory/disk transport."""

    def __init__(self, cfg: Any, global_data_root: str, seed: int, hf_cfg: Any) -> None:
        super().__init__(cfg=cfg, global_data_root=global_data_root, seed=seed)

        self.cfg_dict = to_plain_dict(cfg)
        self.hf_cfg = to_plain_dict({"hf": hf_cfg})["hf"]

        self.name = str(self.cfg_dict["name"])
        self.logger = logging.getLogger(f"dataset.{self.name}")

        self.dataset_root = ensure_dir(Path(global_data_root) / self.name)
        self.meta_path = self.dataset_root / "meta.json"

        cache_cfg = self.cfg_dict.get("cache", {})
        self.transport_mode = str(cache_cfg.get("transport", "memory")).strip().lower()
        if self.transport_mode not in {"memory", "disk"}:
            raise ValueError(f"Unsupported cache.transport for dataset '{self.name}': {self.transport_mode}")
        self.num_chunks_kept = int(cache_cfg.get("num_chunks_kept", 2))
        self.memory_queue_max_chunks = max(1, int(cache_cfg.get("memory_queue_max_chunks", self.num_chunks_kept)))
        self.chunk_cache: ChunkCache | None = None
        if self.transport_mode == "disk":
            self.chunk_cache = ChunkCache(self.dataset_root, num_chunks_kept=self.num_chunks_kept)
            self.chunk_cache.cleanup_incomplete_chunks()

        worker_cfg = self.cfg_dict.get("worker", {})
        self.get_timeout_s = float(worker_cfg.get("get_timeout_s", 5.0))
        self.startup_get_timeout_s = float(worker_cfg.get("startup_get_timeout_s", max(300.0, self.get_timeout_s * 20)))
        self.idle_sleep_s = float(worker_cfg.get("idle_sleep_s", 0.2))
        self.max_worker_restarts = int(worker_cfg.get("max_worker_restarts", 5))

        self._ctx = mp.get_context("spawn")
        self._events_queue: Any | None = None
        self._data_queue: Any | None = None
        self._stop_event: Any | None = None
        self._worker: mp.Process | None = None
        self._worker_restarts = 0
        self._worker_permanently_stopped = False
        self._last_worker_error: str | None = None

        self._known_chunks: set[str] = set()
        self._chunk_queue: deque[dict[str, Any]] = deque()
        self._memory_samples: deque[ImageSample] = deque()
        self._served_samples = 0
        self._status_started_at = float(time.time())
        self._worker_started_at: float | None = None
        self._chunks_ready_total = 0
        self._chunks_evicted_total = 0
        self._records_materialized_total = 0
        self._chunk_build_time_total_s = 0.0
        self._last_chunk_ready_at: float | None = None

        self._init_meta_if_missing()
        if self.chunk_cache is not None:
            self._load_existing_chunks()

    def start(self) -> None:
        self._spawn_worker(force=True)

    def get_batch(self, batch_size: int) -> list[ImageSample]:
        if batch_size <= 0:
            return []

        timeout_s = self.get_timeout_s
        # First chunk warmup can be slower (metadata/parquet/network); avoid false empty batches at startup.
        if self._served_samples == 0 and not self._chunk_queue:
            timeout_s = max(timeout_s, self.startup_get_timeout_s)
        elif self.transport_mode == "memory":
            timeout_s = min(timeout_s, 0.2)

        deadline = time.monotonic() + timeout_s
        batch: list[ImageSample] = []

        while len(batch) < batch_size:
            self._drain_worker_events()
            self._drain_worker_data()
            sample = self._pop_sample()
            if sample is not None:
                batch.append(sample)
                continue

            self._ensure_worker_alive()
            if self._worker_permanently_stopped and not self._chunk_queue:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(self.idle_sleep_s)

        if len(batch) < batch_size:
            if self._last_worker_error:
                self.logger.warning(
                    "Requested %s samples, returned %s (cache underfilled). Last worker error: %s",
                    batch_size,
                    len(batch),
                    self._last_worker_error,
                )
            else:
                self.logger.warning(
                    "Requested %s samples, returned %s (cache currently underfilled)",
                    batch_size,
                    len(batch),
                )

        return batch

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=2)

        self._worker = None

        if self._data_queue is not None:
            self._close_data_queue()

        if self._events_queue is not None:
            try:
                self._events_queue.close()
            except Exception:
                pass
            self._events_queue = None

    def _close_data_queue(self) -> None:
        if self._data_queue is None:
            return
        while True:
            try:
                payload = self._data_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
            if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "image_ref" in payload[0]:
                cleanup_shared_image_rows(payload)
        try:
            self._data_queue.close()
        except Exception:
            pass
        self._data_queue = None

    def stats(self) -> dict[str, Any]:
        self._drain_worker_events()
        self._ensure_worker_alive()
        now = float(time.time())
        total_uptime_s = max(1e-6, now - self._status_started_at)

        worker_pid = None
        worker_exitcode = None
        if self._worker is not None:
            try:
                worker_pid = int(self._worker.pid) if self._worker.pid is not None else None
            except Exception:
                worker_pid = None
            try:
                worker_exitcode = self._worker.exitcode
            except Exception:
                worker_exitcode = None

        return {
            "dataset": self.name,
            "dataset_root": str(self.dataset_root),
            "transport_mode": self.transport_mode,
            "chunks_on_disk": 0 if self.chunk_cache is None else len(self.chunk_cache.list_chunks()),
            "chunks_loaded": len(self._chunk_queue) if self.transport_mode == "disk" else 0,
            "images_on_disk": 0 if self.chunk_cache is None else self.chunk_cache.count_images(),
            "memory_samples_loaded": len(self._memory_samples),
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "worker_pid": worker_pid,
            "worker_exitcode": worker_exitcode,
            "worker_restarts": self._worker_restarts,
            "worker_permanently_stopped": bool(self._worker_permanently_stopped),
            "worker_last_error": self._last_worker_error,
            "samples_served": self._served_samples,
            "chunks_ready_total": int(self._chunks_ready_total),
            "chunks_evicted_total": int(self._chunks_evicted_total),
            "records_materialized": int(self._records_materialized_total),
            "chunk_build_time_total_s": float(self._chunk_build_time_total_s),
            "samples_per_second": float(self._served_samples / total_uptime_s),
            "chunk_ready_rate_per_second": float(self._chunks_ready_total / total_uptime_s),
            "avg_chunk_build_seconds": float(self._chunk_build_time_total_s / max(1, int(self._chunks_ready_total))),
            "worker_uptime_s": float(0.0 if self._worker_started_at is None else max(0.0, now - self._worker_started_at)),
            "last_chunk_ready_at": self._last_chunk_ready_at,
        }

    def _spawn_worker(self, force: bool) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        if self._worker_restarts >= self.max_worker_restarts and not force:
            return

        if self._events_queue is not None:
            try:
                self._events_queue.close()
            except Exception:
                pass
        if self._data_queue is not None:
            self._close_data_queue()

        self._events_queue = self._ctx.Queue(maxsize=512)
        if self.transport_mode == "memory":
            self._data_queue = self._ctx.Queue(maxsize=self.memory_queue_max_chunks)
        else:
            self._data_queue = None
        self._stop_event = self._ctx.Event()

        worker_payload = {
            "dataset_cfg": self.cfg_dict,
            "hf_cfg": self.hf_cfg,
            "global_data_root": str(self.global_data_root),
            "seed": int(self.seed),
        }

        self._worker = self._ctx.Process(
            target=hf_dataset_prefetch_worker_entry,
            args=(worker_payload, self._events_queue, self._data_queue, self._stop_event),
            daemon=True,
            name=f"{self.name}_prefetch",
        )
        self._worker.start()
        self._worker_permanently_stopped = False
        self.logger.info("Started worker pid=%s", self._worker.pid)

    def _ensure_worker_alive(self) -> None:
        if self._worker is None:
            self._spawn_worker(force=True)
            return

        if self._worker.is_alive():
            return

        if self._worker_restarts >= self.max_worker_restarts:
            if not self._worker_permanently_stopped:
                exit_reason = _format_process_exit(self._worker.exitcode)
                self.logger.error("Worker exited unexpectedly: %s", exit_reason)
                self.logger.error("Max worker restarts reached: %s", self.max_worker_restarts)
                self._last_worker_error = f"worker process terminated: {exit_reason}"
            self._worker_permanently_stopped = True
            return

        exit_reason = _format_process_exit(self._worker.exitcode)
        self.logger.error("Worker exited unexpectedly: %s", exit_reason)
        self._last_worker_error = f"worker process terminated: {exit_reason}"
        self._worker_restarts += 1
        self._spawn_worker(force=True)

    def _drain_worker_events(self) -> None:
        if self._events_queue is None:
            return

        while True:
            try:
                event = self._events_queue.get_nowait()
            except Empty:
                break

            event_type = event.get("type")
            if event_type == "chunk_ready":
                self._chunks_ready_total += 1
                records_count = event.get("records_count")
                if records_count is not None:
                    try:
                        self._records_materialized_total += max(0, int(records_count))
                    except Exception:
                        pass
                build_seconds = event.get("build_seconds")
                if build_seconds is not None:
                    try:
                        self._chunk_build_time_total_s += max(0.0, float(build_seconds))
                    except Exception:
                        pass
                self._last_chunk_ready_at = float(time.time())
                if self.transport_mode == "disk":
                    self._register_chunk(str(event["chunk_id"]))
            elif event_type == "chunk_evicted":
                self._chunks_evicted_total += 1
                if self.transport_mode == "disk":
                    self._forget_chunk(str(event["chunk_id"]))
            elif event_type == "worker_started":
                timestamp = event.get("timestamp")
                if timestamp is not None:
                    try:
                        self._worker_started_at = float(timestamp)
                    except Exception:
                        self._worker_started_at = float(time.time())
                else:
                    self._worker_started_at = float(time.time())
            elif event_type == "error":
                message = str(event.get("message", "unknown worker error"))
                self._last_worker_error = message
                self.logger.error("Worker error: %s", message)
            else:
                self.logger.debug("Worker event: %s", event)

    def _drain_worker_data(self) -> None:
        if self._data_queue is None:
            return

        while True:
            try:
                chunk_payload = self._data_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

            if not isinstance(chunk_payload, list):
                continue

            if chunk_payload and isinstance(chunk_payload[0], dict) and "image_ref" in chunk_payload[0]:
                try:
                    chunk_payload = restore_image_rows(chunk_payload, release=True)
                except Exception as exc:
                    self.logger.warning("Failed to restore shared-memory data chunk for dataset=%s: %s", self.name, exc)
                    continue

            for row in chunk_payload:
                if not isinstance(row, dict):
                    continue
                image = row.get("image")
                if image is None:
                    continue
                self._memory_samples.append(
                    ImageSample(
                        image=image,
                        dataset_name=self.name,
                        sample_id=row.get("sample_id", "unknown"),
                        meta=row.get("meta", {}) if isinstance(row.get("meta"), dict) else {},
                    )
                )

    def _register_chunk(self, chunk_id: str) -> None:
        if self.chunk_cache is None:
            return
        if chunk_id in self._known_chunks:
            return

        try:
            manifest = self.chunk_cache.load_chunk(chunk_id)
        except FileNotFoundError:
            # Stale chunk event: chunk may have been evicted before we processed event.
            self.logger.debug("Skipping stale chunk event for missing chunk: %s", chunk_id)
            return
        except Exception as exc:
            self.logger.warning("Cannot load chunk %s: %s", chunk_id, exc)
            return

        items = manifest.get("items", [])
        if not items:
            return

        self._chunk_queue.append(
            {
                "chunk_id": chunk_id,
                "items": items,
                "cursor": 0,
            }
        )
        self._known_chunks.add(chunk_id)

    def _forget_chunk(self, chunk_id: str) -> None:
        self._known_chunks.discard(chunk_id)
        self._chunk_queue = deque(state for state in self._chunk_queue if state.get("chunk_id") != chunk_id)

    def _load_existing_chunks(self) -> None:
        if self.chunk_cache is None:
            return
        for chunk_id in self.chunk_cache.list_chunks():
            self._register_chunk(chunk_id)

    def _pop_sample(self) -> ImageSample | None:
        if self.transport_mode == "memory":
            if not self._memory_samples:
                return None
            sample = self._memory_samples.popleft()
            self._served_samples += 1
            return sample

        while self._chunk_queue:
            state = self._chunk_queue[0]
            chunk_id = str(state["chunk_id"])
            items = state["items"]
            cursor = int(state["cursor"])

            if cursor >= len(items):
                self._consume_chunk(chunk_id)
                continue

            entry = items[cursor]
            state["cursor"] = cursor + 1

            file_name = entry.get("file")
            if not file_name:
                continue

            image_path = self.chunk_cache.chunk_path(chunk_id) / str(file_name)
            if not image_path.exists():
                continue

            try:
                image = load_image(image_path)
            except Exception as exc:
                self.logger.warning("Failed to load image %s: %s", image_path, exc)
                continue

            if state["cursor"] >= len(items):
                self._consume_chunk(chunk_id)

            sample_id = entry.get("sample_id", "unknown")
            meta = entry.get("meta", {})

            self._served_samples += 1
            return ImageSample(
                image=image,
                dataset_name=self.name,
                sample_id=sample_id,
                meta=meta if isinstance(meta, dict) else {},
            )

        return None

    def _consume_chunk(self, chunk_id: str) -> None:
        self._forget_chunk(chunk_id)
        if self.chunk_cache is not None:
            self.chunk_cache.remove_chunk(chunk_id)

    def _init_meta_if_missing(self) -> None:
        if self.meta_path.exists():
            return

        dataset_size: int | None = None
        try:
            token = str(self.hf_cfg.get("token")) if self.hf_cfg.get("token") else None
            if not bool(self.cfg_dict.get("hf", {}).get("streaming", False)):
                dataset = load_hf_dataset(self.cfg_dict, token=token, seed=int(self.seed))
                dataset_size = len(dataset)  # type: ignore[arg-type]
        except Exception as exc:
            self.logger.warning("Could not infer dataset size for meta.json: %s", exc)

        hf_local = self.cfg_dict.get("hf", {})
        schema = self.cfg_dict.get("schema", {})

        payload = {
            "dataset_name": self.name,
            "hf_repo": hf_local.get("repo"),
            "hf_subset": hf_local.get("subset"),
            "hf_split": hf_local.get("split"),
            "schema": {
                "image_mode": schema.get("image_mode"),
                "image_field": schema.get("image_field"),
                "url_field": schema.get("url_field"),
                "extra_fields": schema.get("extra_fields", []),
            },
            "streaming": bool(hf_local.get("streaming", False)),
            "trust_remote_code": bool(hf_local.get("trust_remote_code", False)),
            "dataset_size": dataset_size,
            "gated": bool(self.cfg_dict.get("gated", False)),
            "licensing_note": self.cfg_dict.get("licensing_note", ""),
            "created_at": utc_now_iso(),
        }
        write_json_atomic(self.meta_path, payload)


def hf_dataset_prefetch_worker(worker_payload: dict[str, Any], events_queue: Any, data_queue: Any, stop_event: Any) -> None:
    dataset_cfg = to_plain_dict(worker_payload["dataset_cfg"])
    forensics_cfg = _build_forensics_cfg(dataset_cfg)
    hf_cfg = to_plain_dict({"hf": worker_payload["hf_cfg"]})["hf"]

    dataset_name = str(dataset_cfg["name"])
    log_path = _configure_logging(dataset_cfg)
    logger = logging.getLogger(f"dataset.worker.{dataset_name}")
    maybe_enable_core_dumps(forensics_cfg, section="train", logger=logger)
    fault_path = _enable_fault_handler_log(log_path, label=f"{dataset_name}_worker", logger=logger)
    if fault_path is not None:
        logger.info("Worker fault log file: %s", fault_path)

    top_cfg = {"hf": hf_cfg}
    allow_missing_token = not bool(dataset_cfg.get("gated", False))
    token = init_hf_auth(top_cfg, allow_missing_token=allow_missing_token)

    global_data_root = str(worker_payload["global_data_root"])
    seed = int(worker_payload["seed"])

    dataset_root = ensure_dir(Path(global_data_root) / dataset_name)
    cache_cfg = dataset_cfg.get("cache", {})
    transport_mode = str(cache_cfg.get("transport", "memory")).strip().lower()
    chunk_size = int(cache_cfg.get("chunk_size_images", 1024))
    num_chunks_kept = int(cache_cfg.get("num_chunks_kept", 2))

    worker_cfg = dataset_cfg.get("worker", {})
    idle_sleep_s = float(worker_cfg.get("idle_sleep_s", 0.5))
    request_timeout_s = int(worker_cfg.get("request_timeout_s", 10))
    max_retries = int(worker_cfg.get("max_retries", 2))
    max_chunk_build_seconds = float(worker_cfg.get("max_chunk_build_seconds", 20.0))

    cache: ChunkCache | None = None
    if transport_mode == "disk":
        cache = ChunkCache(dataset_root=dataset_root, num_chunks_kept=num_chunks_kept)
        cleaned = cache.cleanup_incomplete_chunks()
        if cleaned:
            logger.info("Removed %s incomplete chunks at worker startup for dataset=%s", len(cleaned), dataset_name)

    rng = random.Random(seed + int(time.time()))

    streaming = bool(dataset_cfg.get("hf", {}).get("streaming", False))
    dataset = None
    dataset_server_sampler: DatasetServerSampler | None = None
    hf_local = dataset_cfg.get("hf", {})
    if streaming and bool(hf_local.get("stream_via_datasets_server", False)):
        try:
            dataset_server_sampler = DatasetServerSampler(
                repo=str(hf_local.get("repo")),
                split=str(hf_local.get("split", "train")),
                subset=hf_local.get("subset"),
                token=token,
                seed=seed,
                timeout_s=max(5.0, float(request_timeout_s)),
                max_retries=max(1, int(max_retries)),
            )
            logger.info(
                "Using datasets-server sampler for dataset=%s config=%s split=%s num_examples=%s",
                dataset_name,
                dataset_server_sampler.config,
                dataset_server_sampler.split,
                dataset_server_sampler.num_examples,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("datasets-server sampler init failed for %s: %s. Falling back to load_dataset streaming.", dataset_name, exc)

    if dataset_server_sampler is None:
        try:
            dataset = load_hf_dataset(dataset_cfg, token=token, seed=seed)
        except Exception as exc:
            message = _format_dataset_load_error(dataset_cfg=dataset_cfg, exc=exc)
            traceback_text = traceback.format_exc()
            report_path = emit_fatal_report(
                forensics_cfg,
                role=f"dataset_worker_{dataset_name}",
                error=str(exc),
                traceback_text=traceback_text,
                extra={
                    "dataset": dataset_name,
                    "phase": "load_hf_dataset",
                },
                section="train",
            )
            logger.exception("Dataset load failed for %s", dataset_name)
            _emit_event(
                events_queue,
                {
                    "type": "error",
                    "dataset": dataset_name,
                    "message": message,
                    "fatal_report_path": report_path,
                },
            )
            return

    dataset_size = _safe_len(dataset) if (dataset is not None and not streaming) else None
    iterator = iter(dataset) if (dataset is not None and streaming) else None
    stream_counter = 0

    logger.info("Worker started for dataset=%s streaming=%s", dataset_name, streaming)
    _emit_event(
        events_queue,
        {
            "type": "worker_started",
            "dataset": dataset_name,
            "pid": int(os.getpid()),
            "timestamp": float(time.time()),
        },
    )

    while not stop_event.is_set():
        try:
            if cache is not None:
                evicted = cache.evict_old_chunks()
                for chunk_id in evicted:
                    _emit_event(events_queue, {"type": "chunk_evicted", "chunk_id": chunk_id, "dataset": dataset_name})

                chunks_now = cache.list_chunks()
                if len(chunks_now) >= num_chunks_kept:
                    time.sleep(idle_sleep_s)
                    continue

            chunk_id = _new_chunk_id(rng) if cache is None else cache.start_chunk(_new_chunk_id(rng))
            records: list[dict[str, Any]] = []
            attempts = 0
            max_attempts = max(chunk_size * 10, chunk_size + 32)
            chunk_started_at = time.monotonic()
            materialize_none_reasons: Counter[str] = Counter()
            materialize_exception_count = 0
            record_fetch_exception_count = 0
            record_none_count = 0

            while len(records) < chunk_size and not stop_event.is_set():
                attempts += 1
                if attempts > max_attempts:
                    logger.warning(
                        "Stopping chunk fill early after %s attempts (%s records collected; "
                        "record_none=%s; record_fetch_exceptions=%s; materialize_exceptions=%s; "
                        "materialize_none_reasons=%s)",
                        attempts,
                        len(records),
                        record_none_count,
                        record_fetch_exception_count,
                        materialize_exception_count,
                        dict(sorted(materialize_none_reasons.items())),
                    )
                    break

                if records and (time.monotonic() - chunk_started_at) >= max_chunk_build_seconds:
                    logger.debug(
                        "Finalizing partial chunk after %.1fs with %s records",
                        time.monotonic() - chunk_started_at,
                        len(records),
                    )
                    break

                try:
                    record, sample_id, iterator, stream_counter = _next_record(
                        dataset=dataset,
                        dataset_cfg=dataset_cfg,
                        rng=rng,
                        dataset_size=dataset_size,
                        iterator=iterator,
                        stream_counter=stream_counter,
                        token=token,
                        seed=seed,
                        dataset_server_sampler=dataset_server_sampler,
                    )
                except Exception as exc:  # noqa: BLE001
                    record_fetch_exception_count += 1
                    logger.warning("Record fetch failed for dataset=%s: %s", dataset_name, exc)
                    time.sleep(max(0.1, idle_sleep_s))
                    continue
                if record is None:
                    record_none_count += 1
                    continue

                try:
                    materialized, materialize_reason = _materialize_image(
                        record=record,
                        sample_id=sample_id,
                        dataset_cfg=dataset_cfg,
                        cache=cache,
                        chunk_id=chunk_id,
                        transport_mode=transport_mode,
                        timeout=request_timeout_s,
                        retries=max_retries,
                    )
                except Exception as exc:  # noqa: BLE001
                    materialize_exception_count += 1
                    logger.warning(
                        "Image materialization failed for dataset=%s sample_id=%s: %s",
                        dataset_name,
                        sample_id,
                        exc,
                    )
                    continue
                if materialized is None:
                    materialize_none_reasons[str(materialize_reason or "unknown")] += 1
                    continue

                image_payload, item_meta = materialized
                if transport_mode == "memory":
                    records.append(
                        {
                            "sample_id": str(sample_id),
                            "image": image_payload,
                            "meta": item_meta,
                        }
                    )
                else:
                    image_path = Path(str(image_payload))
                    records.append(
                        {
                            "sample_id": str(sample_id),
                            "file": image_path.name,
                            "meta": item_meta,
                        }
                    )

            if records:
                if cache is not None:
                    cache.finalize_chunk(chunk_id, records)
                else:
                    _emit_data_chunk(data_queue, records, stop_event=stop_event)
                _emit_event(
                    events_queue,
                    {
                        "type": "chunk_ready",
                        "chunk_id": chunk_id,
                        "dataset": dataset_name,
                        "records_count": int(len(records)),
                        "build_seconds": float(max(0.0, time.monotonic() - chunk_started_at)),
                    },
                )
            else:
                if cache is not None:
                    cache.remove_chunk(chunk_id)
                time.sleep(idle_sleep_s)
        except Exception as exc:
            logger.exception("Worker loop failure for dataset=%s", dataset_name)
            _emit_event(events_queue, {"type": "error", "dataset": dataset_name, "message": str(exc)})
            time.sleep(max(1.0, idle_sleep_s))


def hf_dataset_prefetch_worker_entry(worker_payload: dict[str, Any], events_queue: Any, data_queue: Any, stop_event: Any) -> None:
    dataset_cfg = to_plain_dict(worker_payload.get("dataset_cfg", {}))
    dataset_name = str(dataset_cfg.get("name", "unknown_dataset"))
    forensics_cfg = _build_forensics_cfg(dataset_cfg)
    maybe_redirect_stdio(
        forensics_cfg,
        role=f"dataset_worker_{dataset_name}",
        section="train",
    )
    logger = logging.getLogger(f"dataset.worker.{dataset_name}")
    maybe_enable_core_dumps(forensics_cfg, section="train", logger=logger)
    try:
        hf_dataset_prefetch_worker(worker_payload, events_queue, data_queue, stop_event)
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        report_path = emit_fatal_report(
            forensics_cfg,
            role=f"dataset_worker_{dataset_name}",
            error=str(exc),
            traceback_text=traceback_text,
            extra={
                "dataset": dataset_name,
                "phase": "worker_entry_unhandled_exception",
            },
            section="train",
        )
        _emit_event(
            events_queue,
            {
                "type": "error",
                "dataset": dataset_name,
                "message": f"fatal_worker_exception: {exc}",
                "fatal_report_path": report_path,
            },
        )
        logger.exception("Dataset worker crashed with unhandled exception")
        raise


def _safe_len(dataset: Any) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except Exception:
        return None


def _next_record(
    dataset: Any | None,
    dataset_cfg: dict[str, Any],
    rng: random.Random,
    dataset_size: int | None,
    iterator: Any,
    stream_counter: int,
    token: str | None,
    seed: int,
    dataset_server_sampler: DatasetServerSampler | None = None,
) -> tuple[dict[str, Any] | None, str | int, Any, int]:
    hf_cfg = dataset_cfg.get("hf", {})
    schema = dataset_cfg.get("schema", {})
    id_field = schema.get("id_field")

    if dataset_server_sampler is not None:
        record, server_sample_id = dataset_server_sampler.next_record()
        stream_counter += 1
        sample_id = _extract_sample_id(record, id_field=id_field, fallback=server_sample_id)
        return record, sample_id, iterator, stream_counter

    if bool(hf_cfg.get("streaming", False)):
        if dataset is None:
            return None, "unknown", iterator, stream_counter
        if iterator is None:
            iterator = iter(dataset)
        try:
            record = next(iterator)
        except StopIteration:
            dataset = load_hf_dataset(dataset_cfg, token=token, seed=seed)
            iterator = iter(dataset)
            record = next(iterator)

        stream_counter += 1
        sample_id = _extract_sample_id(record, id_field=id_field, fallback=f"stream_{stream_counter}")
        return record, sample_id, iterator, stream_counter

    if not dataset_size:
        return None, "unknown", iterator, stream_counter

    random_index = rng.randrange(dataset_size)
    record = dataset[random_index]
    sample_id = _extract_sample_id(record, id_field=id_field, fallback=random_index)
    return record, sample_id, iterator, stream_counter


def _extract_sample_id(record: dict[str, Any], id_field: str | None, fallback: str | int) -> str | int:
    if id_field and id_field in record:
        return record[id_field]

    for candidate in ("id", "image_id", "sample_id", "key"):
        if candidate in record:
            return record[candidate]

    return fallback


def _materialize_image(
    record: dict[str, Any],
    sample_id: str | int,
    dataset_cfg: dict[str, Any],
    cache: ChunkCache | None,
    chunk_id: str | None,
    transport_mode: str,
    timeout: int,
    retries: int,
) -> tuple[tuple[Any, dict[str, Any]] | None, str]:
    schema = dataset_cfg.get("schema", {})
    mode = str(schema.get("image_mode", "image_field"))

    image_payload: Any | None = None

    if mode == "image_field":
        field = _resolve_image_field(record, schema)
        if field is None:
            return None, "missing_image_field"
        image_value = record[field]
        image_payload = _materialize_image_field_value(
            image_value=image_value,
            sample_id=sample_id,
            cache=cache,
            chunk_id=chunk_id,
            transport_mode=transport_mode,
            timeout=timeout,
            retries=retries,
        )
        if image_payload is None:
            return None, "image_field_decode_or_fetch_failed"
    elif mode == "url_field":
        url = _resolve_url(record, schema)
        if not url:
            return None, "missing_url"
        if transport_mode == "memory":
            image_payload = fetch_image_to_memory(
                url=url,
                timeout=timeout,
                retries=retries,
            )
        else:
            if cache is None or chunk_id is None:
                raise RuntimeError("disk transport requires active ChunkCache and chunk_id")
            image_payload = fetch_image_to_cache(
                url=url,
                cache=cache,
                sample_key=sample_id,
                timeout=timeout,
                retries=retries,
                chunk_id=chunk_id,
            )
        if image_payload is None:
            return None, "url_fetch_failed"
    else:
        raise ValueError(f"Unsupported schema.image_mode: {mode}")

    if image_payload is None:
        return None, "unknown"

    item_meta: dict[str, Any] = {}
    for extra_key in schema.get("extra_fields", []):
        if extra_key in record:
            item_meta[extra_key] = _safe_meta_value(record[extra_key])

    return (image_payload, item_meta), "ok"


def _materialize_image_field_value(
    image_value: Any,
    sample_id: str | int,
    cache: ChunkCache | None,
    chunk_id: str | None,
    transport_mode: str,
    timeout: int,
    retries: int,
) -> Any | None:
    # `datasets.Image(decode=False)` may produce dict payloads with `bytes`, `path` or `src`.
    if isinstance(image_value, dict):
        bytes_value = image_value.get("bytes")
        if bytes_value is not None:
            image = decode_to_pil(bytes_value)
            return _store_materialized_image(
                image=image,
                sample_id=sample_id,
                cache=cache,
                chunk_id=chunk_id,
                transport_mode=transport_mode,
            )

        for key in ("path", "src", "url"):
            url_like = image_value.get(key)
            if isinstance(url_like, str) and url_like.startswith(("http://", "https://")):
                if transport_mode == "memory":
                    return fetch_image_to_memory(
                        url=url_like,
                        timeout=timeout,
                        retries=retries,
                    )
                if cache is None or chunk_id is None:
                    raise RuntimeError("disk transport requires active ChunkCache and chunk_id")
                return fetch_image_to_cache(
                    url=url_like,
                    cache=cache,
                    sample_key=sample_id,
                    timeout=timeout,
                    retries=retries,
                    chunk_id=chunk_id,
                )

        path_value = image_value.get("path")
        if isinstance(path_value, str) and path_value:
            image = decode_to_pil({"path": path_value})
            return _store_materialized_image(
                image=image,
                sample_id=sample_id,
                cache=cache,
                chunk_id=chunk_id,
                transport_mode=transport_mode,
            )

    if isinstance(image_value, str) and image_value.startswith(("http://", "https://")):
        if transport_mode == "memory":
            return fetch_image_to_memory(
                url=image_value,
                timeout=timeout,
                retries=retries,
            )
        if cache is None or chunk_id is None:
            raise RuntimeError("disk transport requires active ChunkCache and chunk_id")
        return fetch_image_to_cache(
            url=image_value,
            cache=cache,
            sample_key=sample_id,
            timeout=timeout,
            retries=retries,
            chunk_id=chunk_id,
        )

    image = decode_to_pil(image_value)
    return _store_materialized_image(
        image=image,
        sample_id=sample_id,
        cache=cache,
        chunk_id=chunk_id,
        transport_mode=transport_mode,
    )


def _store_materialized_image(
    image: Any,
    sample_id: str | int,
    cache: ChunkCache | None,
    chunk_id: str | None,
    transport_mode: str,
) -> Any:
    if transport_mode == "memory":
        return image
    if cache is None or chunk_id is None:
        raise RuntimeError("disk transport requires active ChunkCache and chunk_id")
    return cache.save_image(sample_id=sample_id, image=image, chunk_id=chunk_id)


def _resolve_image_field(record: dict[str, Any], schema: dict[str, Any]) -> str | None:
    primary = schema.get("image_field")
    candidates = [primary, *schema.get("image_field_candidates", []), "image", "img"]

    for candidate in candidates:
        if candidate and candidate in record:
            return str(candidate)

    return None


def _resolve_url(record: dict[str, Any], schema: dict[str, Any]) -> str | None:
    primary = schema.get("url_field")
    candidates = [
        primary,
        *schema.get("url_field_candidates", []),
        "url",
        "image_url",
        "imageURL",
        "jpg",
    ]

    for candidate in candidates:
        if not candidate or candidate not in record:
            continue
        value = record[candidate]
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value

    return None


def _safe_meta_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_meta_value(item) for item in value[:32]]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in list(value.items())[:32]:
            compact[str(key)] = _safe_meta_value(item)
        return compact
    return str(value)


def _new_chunk_id(rng: random.Random) -> str:
    return f"chunk_{int(time.time() * 1000)}_{rng.randint(1000, 9999)}"


def _emit_event(queue_obj: Any, payload: dict[str, Any]) -> None:
    try:
        queue_obj.put_nowait(payload)
    except Full:
        pass


def _emit_data_chunk(queue_obj: Any, payload: list[dict[str, Any]], *, stop_event: Any) -> None:
    if queue_obj is None:
        return
    shared_payload = share_image_rows(payload)
    enqueued = False
    try:
        while not stop_event.is_set():
            try:
                queue_obj.put(shared_payload, timeout=0.5)
                enqueued = True
                return
            except Full:
                continue
    finally:
        if not enqueued:
            cleanup_shared_image_rows(shared_payload)


def _format_dataset_load_error(dataset_cfg: dict[str, Any], exc: Exception) -> str:
    if not bool(dataset_cfg.get("gated", False)):
        return str(exc)

    base = (
        f"{exc}. Set hf.token in conf/config.yaml. "
        "Accept the dataset license on Hugging Face dataset page. "
        "Ensure token has access (for fine-grained tokens enable access to public gated repositories)."
    )
    return base


def _configure_logging(dataset_cfg: dict[str, Any]) -> Path:
    runtime_cfg = {
        "data": {"log_level": str(dataset_cfg.get("log_level", "INFO"))},
        "logging": dataset_cfg.get("runtime_logging", {}),
    }
    return configure_root_logging(cfg=runtime_cfg, rank=0, force=True)
