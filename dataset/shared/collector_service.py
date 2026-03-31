from __future__ import annotations

import faulthandler
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import traceback
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Full, Queue as ThreadQueue
from typing import Any

from omegaconf import DictConfig

from dataset.data_raw.core.config import to_plain_dict
from dataset.data_raw.core.types import ImageSampleRef
from dataset.logging_utils import configure_process_logging
from dataset.models.model_pool import ModelPool
from dataset.shared.atomizer import atomize
from dataset.shared.cache import SharedSampleCache
from dataset.shared.compatibility_index import (
    CompatibilityIndex,
    resolve_collector_device,
    resolve_collector_mode,
    resolve_train_device,
)
from .load_report import LoadReportWriter
from dataset.shared.model_scheduler import ModelScheduler
from dataset.shared.raw_dataset_pool import RawDatasetPool
from dataset.shared.shm_transport import cleanup_shared_layer_record_refs, restore_layer_record_refs, share_layer_record_refs
from dataset.shared.streaming.factory import (
    build_chunk_reader,
    build_chunk_store,
    build_chunk_writer,
    resolve_streaming_cfg,
)
from dataset.shared.types import CollectorJobStats, SharedSample
from training.forensics import emit_fatal_report, maybe_enable_core_dumps, maybe_redirect_stdio

_FAULT_HANDLER_FILES: list[Any] = []


@dataclass(slots=True)
class _PreparedCollectorJob:
    model_name: str
    image_refs: list[ImageSampleRef]
    image_meta: list[Any]
    dataset_mix: dict[str, int]
    started_at: float
    raw_batch_fetch_s: float


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_proc_status_fields(pid: int) -> dict[str, str]:
    path = Path(f"/proc/{int(pid)}/status")
    text = _read_text_file(path)
    if not text:
        return {}
    payload: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _parse_kb_to_mb(raw_value: str | None) -> float | None:
    if not raw_value:
        return None
    token = str(raw_value).strip().split(" ", 1)[0]
    if not token:
        return None
    try:
        return float(int(token) / 1024.0)
    except Exception:
        return None


def _list_child_pids(pid: int) -> list[int]:
    children_path = Path(f"/proc/{int(pid)}/task/{int(pid)}/children")
    text = _read_text_file(children_path)
    if not text:
        return []
    out: list[int] = []
    for token in text.strip().split():
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def _collect_process_resource_snapshot(*, pid: int | None = None, max_children: int = 8) -> dict[str, Any]:
    target_pid = int(os.getpid() if pid is None else pid)
    payload: dict[str, Any] = {"pid": target_pid}

    proc_fields = _read_proc_status_fields(target_pid)
    if proc_fields:
        payload["name"] = proc_fields.get("Name")
        payload["state"] = proc_fields.get("State")
        payload["threads"] = int(proc_fields.get("Threads", "0") or 0)
        payload["rss_mb"] = _parse_kb_to_mb(proc_fields.get("VmRSS"))
        payload["hwm_mb"] = _parse_kb_to_mb(proc_fields.get("VmHWM"))
        payload["vms_mb"] = _parse_kb_to_mb(proc_fields.get("VmSize"))
        payload["swap_mb"] = _parse_kb_to_mb(proc_fields.get("VmSwap"))

    fd_dir = Path(f"/proc/{target_pid}/fd")
    try:
        payload["open_fds"] = len(list(fd_dir.iterdir()))
    except Exception:
        payload["open_fds"] = None

    children = _list_child_pids(target_pid)
    payload["children_count"] = int(len(children))
    child_rows: list[dict[str, Any]] = []
    children_rss_mb = 0.0
    for child_pid in children[: max(0, int(max_children))]:
        child_fields = _read_proc_status_fields(child_pid)
        child_rss = _parse_kb_to_mb(child_fields.get("VmRSS")) if child_fields else None
        if child_rss is not None:
            children_rss_mb += float(child_rss)
        child_rows.append(
            {
                "pid": int(child_pid),
                "name": child_fields.get("Name") if child_fields else None,
                "state": child_fields.get("State") if child_fields else None,
                "rss_mb": child_rss,
                "threads": int(child_fields.get("Threads", "0") or 0) if child_fields else None,
            }
        )
    payload["children_rss_mb_sum"] = float(children_rss_mb)
    if child_rows:
        payload["children_head"] = child_rows
    return payload


def _read_cgroup_scalar(paths: list[str]) -> int | str | None:
    for path_str in paths:
        text = _read_text_file(Path(path_str))
        if not text:
            continue
        token = text.strip().splitlines()[0].strip()
        if not token:
            continue
        if token == "max":
            return token
        try:
            return int(token)
        except Exception:
            continue
    return None


def _read_cgroup_memory_events() -> dict[str, int]:
    candidates = [
        "/sys/fs/cgroup/memory.events",
        "/sys/fs/cgroup/memory/memory.events",
    ]
    for path_str in candidates:
        text = _read_text_file(Path(path_str))
        if not text:
            continue
        out: dict[str, int] = {}
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            key, raw_value = parts
            try:
                out[str(key)] = int(raw_value)
            except Exception:
                continue
        if out:
            return out
    return {}


def _read_cgroup_memory_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "events": _read_cgroup_memory_events(),
        "memory_current_bytes": _read_cgroup_scalar(
            ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]
        ),
        "memory_max_bytes": _read_cgroup_scalar(
            ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
        ),
        "memory_swap_current_bytes": _read_cgroup_scalar(
            ["/sys/fs/cgroup/memory.swap.current", "/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes"]
        ),
        "memory_swap_max_bytes": _read_cgroup_scalar(
            ["/sys/fs/cgroup/memory.swap.max", "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"]
        ),
    }
    # Remove empty keys to keep status compact.
    return {k: v for k, v in payload.items() if v not in (None, {}, "")}


def _diff_counter_dict(current: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in current.items():
        try:
            out[str(key)] = int(value) - int(baseline.get(key, 0))
        except Exception:
            continue
    return out


def _write_json_report(path: Path, payload: dict[str, Any], logger: logging.Logger) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to write collector crash report '%s': %s", path, exc)


def _append_jsonl_record(path: Path, payload: dict[str, Any], logger: logging.Logger) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("Failed to append JSONL record '%s': %s", path, exc)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _annotate_dataset_workers_with_resource(
    raw_pool_workers: dict[str, Any] | None,
    resource_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(raw_pool_workers, dict):
        return {}

    children_by_pid: dict[int, dict[str, Any]] = {}
    if isinstance(resource_payload, dict):
        children = resource_payload.get("children_head")
        if isinstance(children, list):
            for row in children:
                if not isinstance(row, dict):
                    continue
                pid = _safe_int(row.get("pid"))
                if pid is None:
                    continue
                children_by_pid[pid] = row

    payload: dict[str, Any] = {}
    for dataset_name, info in raw_pool_workers.items():
        if isinstance(info, dict):
            row = dict(info)
        else:
            row = {"value": info}

        worker_pid = _safe_int(row.get("worker_pid"))
        if worker_pid is not None:
            child = children_by_pid.get(worker_pid)
            if isinstance(child, dict):
                row["worker_rss_mb"] = child.get("rss_mb")
                row["worker_state"] = child.get("state")
                row["worker_threads"] = child.get("threads")
                row["worker_name"] = child.get("name")
        payload[str(dataset_name)] = row

    return payload


def _top_worker_rss(
    dataset_workers: dict[str, Any] | None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not isinstance(dataset_workers, dict):
        return []

    rows: list[dict[str, Any]] = []
    for dataset_name, info in dataset_workers.items():
        if not isinstance(info, dict):
            continue
        rss_mb = _safe_float(info.get("worker_rss_mb"))
        if rss_mb is None:
            continue
        rows.append(
            {
                "dataset": str(dataset_name),
                "worker_pid": _safe_int(info.get("worker_pid")),
                "worker_rss_mb": float(rss_mb),
                "worker_state": info.get("worker_state"),
                "worker_last_error": info.get("worker_last_error"),
            }
        )

    rows.sort(key=lambda item: float(item.get("worker_rss_mb", 0.0)), reverse=True)
    return rows[: max(1, int(limit))]


def _top_jobs_by_memory_field(
    jobs: list[dict[str, Any]] | None,
    *,
    field: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(jobs, list):
        return []

    rows: list[dict[str, Any]] = []
    for row in jobs:
        if not isinstance(row, dict):
            continue
        value = _safe_float(row.get(field))
        if value is None:
            continue
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "model_name": row.get("model_name"),
                "duration_s": _safe_float(row.get("duration_s")),
                "num_images": _safe_int(row.get("num_images")),
                "num_layers": _safe_int(row.get("num_layers")),
                "num_samples_emitted": _safe_int(row.get("num_samples_emitted")),
                "field": field,
                "value": float(value),
            }
        )
    rows.sort(key=lambda item: float(item.get("value", 0.0)), reverse=True)
    return rows[: max(1, int(limit))]


def _extract_job_memory_markers(resource_snapshot: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(resource_snapshot, dict):
        return {
            "rss_mb": None,
            "hwm_mb": None,
            "children_rss_mb_sum": None,
        }
    return {
        "rss_mb": _safe_float(resource_snapshot.get("rss_mb")),
        "hwm_mb": _safe_float(resource_snapshot.get("hwm_mb")),
        "children_rss_mb_sum": _safe_float(resource_snapshot.get("children_rss_mb_sum")),
    }


def _delta_or_none(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after - before)


def _fmt_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


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
        logger.warning("Failed to initialize fault-handler log file: %s", exc)
        return None


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

        self.collector_device = resolve_collector_device(self.cfg_dict)
        self.train_device = resolve_train_device(self.cfg_dict)
        allow_async_on_train_device = bool(collector_cfg.get("allow_async_on_train_device", False))
        self.parallel_inference_workers = max(1, int(collector_cfg.get("parallel_inference_workers", 1)))
        self.parallel_model_pool_max_loaded_models = max(
            1,
            int(collector_cfg.get("parallel_model_pool_max_loaded_models", 1)),
        )
        self.prepared_job_queue_max_items = max(
            self.parallel_inference_workers,
            int(collector_cfg.get("prepared_job_queue_max_items", max(4, self.parallel_inference_workers * 2))),
        )
        self.atomizer_process_enabled = bool(collector_cfg.get("atomizer_process_enabled", False))
        self.atomizer_queue_max_items = max(8, int(collector_cfg.get("atomizer_queue_max_items", 256)))

        if self.collector_mode == "async":
            if self.collector_device is None:
                raise ValueError(
                    "collector.device (or collector.device_candidates) must resolve to a device for async collector mode"
                )
            if self.train_device is not None and self.collector_device == self.train_device and not allow_async_on_train_device:
                raise ValueError(
                    "async collector mode requires collector.device != train.device "
                    "(unless collector.allow_async_on_train_device=true)"
                )
        if self.parallel_inference_workers > 1 and self.collector_mode != "async":
            raise ValueError("collector.parallel_inference_workers > 1 is supported only in async collector mode")
        if self.parallel_inference_workers > 1 and self.streaming_mode != "none":
            raise ValueError(
                "collector.parallel_inference_workers > 1 is currently supported only with streaming.mode=none"
            )

        self.model_selection_strategy = str(collector_cfg.get("model_selection_strategy", "round_robin_shuffled"))
        self.jobs_per_selected_model = int(collector_cfg.get("jobs_per_selected_model", 2))
        self.dataset_sampling_strategy = str(collector_cfg.get("dataset_sampling_strategy", "weighted_random"))
        self.max_dataset_fraction_per_batch = float(collector_cfg.get("max_dataset_fraction_per_batch", 0.6))
        self.num_inflight_jobs = max(1, int(collector_cfg.get("num_inflight_jobs", 2)))
        self.status_emit_interval_s = max(0.1, float(collector_cfg.get("status_emit_interval_s", 2.0)))
        self.status_queue_max_items = max(16, int(collector_cfg.get("status_queue_max_items", 2048)))
        if self.atomizer_process_enabled and self.streaming_mode != "none":
            raise ValueError("collector.atomizer_process_enabled is currently supported only with streaming.mode=none")
        diagnostics_cfg = collector_cfg.get("diagnostics", {})
        self.emit_resource_snapshot = bool(diagnostics_cfg.get("emit_resource_snapshot", True))
        self.resource_snapshot_max_children = max(0, int(diagnostics_cfg.get("resource_snapshot_max_children", 8)))
        self.include_cgroup_on_failure = bool(diagnostics_cfg.get("include_cgroup_on_failure", True))
        self.crash_report_enabled = bool(diagnostics_cfg.get("crash_report_enabled", True))
        self.crash_report_path = Path(
            str(diagnostics_cfg.get("crash_report_path", "./data/reports/collector_crash_report.json"))
        )
        self.worker_status_enabled = bool(diagnostics_cfg.get("worker_status_enabled", True))
        self.worker_status_dir = Path(
            str(diagnostics_cfg.get("worker_status_dir", "./data/reports/dataset_workers"))
        )

        self.interleaved_cfg = collector_cfg.get("interleaved_schedule", {})
        collect_every_n_steps_cfg = self.interleaved_cfg.get(
            "collect_every_n_train_steps",
            self.interleaved_cfg.get("collect_every_n_steps", 30),
        )
        self.interleaved_every_n_steps = max(
            1,
            int(collect_every_n_steps_cfg),
        )
        self.interleaved_burst_jobs = max(
            1,
            int(self.interleaved_cfg.get("collector_jobs_per_cycle", 1)),
        )

        self.atom_cfg = collector_cfg.get("layer_output_splitting", {})
        self._validate_layer_output_splitting(self.atom_cfg)

        max_loaded = int(collector_cfg.get("max_loaded_models", 1))
        if max_loaded < 1:
            raise ValueError("collector.max_loaded_models must be >= 1")

        self._runtime_ready = False
        self._raw_pool: RawDatasetPool | None = None
        self._model_pool: ModelPool | None = None
        self._scheduler: ModelScheduler | None = None
        self._sink: _SampleSink | None = None
        self._load_report = LoadReportWriter(self.cfg_dict)
        self._parallel_executor: ThreadPoolExecutor | None = None
        self._parallel_model_pools: dict[int, ModelPool] = {}
        self._parallel_model_pools_lock = threading.Lock()
        self._prepared_job_queue: ThreadQueue[_PreparedCollectorJob] | None = None
        self._prepared_job_stop_event: threading.Event | None = None
        self._prepared_job_producer_thread: threading.Thread | None = None
        self._prepared_job_producer_error: dict[str, str] | None = None
        self._atomizer_task_queue: Any | None = None
        self._atomizer_stop_event: Any | None = None
        self._atomizer_process: mp.Process | None = None
        self._stats_lock = threading.Lock()
        self._runtime_device: str | None = None
        self._runtime_local_only = False
        self._runtime_release_device_on_unload = True
        self._runtime_empty_cuda_cache_on_unload = True

        self._model_run_id = 0
        self._jobs_total = 0
        self._jobs_by_model: Counter[str] = Counter()
        self._items_emitted = 0
        self._failed_models: set[str] = set()

        self._ctx = mp.get_context("spawn")
        self._stop_event: Any | None = None
        self._process: mp.Process | None = None
        self._status_queue: Any | None = status_queue
        self._last_status_emit_ts = 0.0
        self._async_last_status: dict[str, Any] | None = None
        self._async_last_atomizer_event: dict[str, Any] | None = None
        self._async_recent_jobs: deque[dict[str, Any]] = deque(maxlen=128)
        self._async_events_dropped = 0
        self._cgroup_events_baseline = _read_cgroup_memory_events()

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
            if isinstance(self._async_last_atomizer_event, dict):
                if self._async_last_atomizer_event.get("type") == "atomizer_fatal_exception":
                    raise RuntimeError(
                        "Async collector atomizer process exited unexpectedly: "
                        f"{self._async_last_atomizer_event.get('fatal_error')}"
                    )
            if isinstance(self._async_last_status, dict):
                if bool(self._async_last_status.get("atomizer_process_enabled", False)) and not bool(
                    self._async_last_status.get("atomizer_process_alive", True)
                ):
                    raise RuntimeError("Async collector atomizer process is no longer alive")
                if self._async_last_status.get("prepared_job_producer_error"):
                    producer_error = self._async_last_status.get("prepared_job_producer_error", {})
                    raise RuntimeError(
                        "Async collector prepared-job producer failed: "
                        f"{producer_error.get('error') if isinstance(producer_error, dict) else producer_error}"
                    )
            return

        exit_code = self._process.exitcode
        exit_reason = _format_process_exit(exit_code)
        hint_parts: list[str] = []
        cgroup_snapshot: dict[str, Any] = {}
        if exit_code == -9:
            hint_parts.append(
                "collector received SIGKILL (often OOM / memory limit). "
                "Try smaller collector load (jobs_per_selected_model=1, num_inflight_jobs=1, "
                "smaller xy_samples_random_slice/chunk_size_samples)."
            )
            if self.include_cgroup_on_failure:
                cgroup_snapshot = _read_cgroup_memory_snapshot()
                cgroup_events = cgroup_snapshot.get("events")
                if isinstance(cgroup_events, dict) and self._cgroup_events_baseline:
                    try:
                        cgroup_snapshot["events_delta"] = _diff_counter_dict(cgroup_events, self._cgroup_events_baseline)
                    except Exception:
                        pass
                if cgroup_snapshot:
                    try:
                        hint_parts.append("cgroup=" + json.dumps(cgroup_snapshot, ensure_ascii=False))
                    except Exception:
                        pass
        elif self.include_cgroup_on_failure:
            cgroup_snapshot = _read_cgroup_memory_snapshot()

        crash_report_payload = self._build_crash_report_payload(
            exit_code=exit_code,
            exit_reason=exit_reason,
            cgroup_snapshot=cgroup_snapshot,
        )
        crash_report_written = self._maybe_write_crash_report(crash_report_payload)
        if crash_report_written is not None:
            hint_parts.append(f"crash_report_path={crash_report_written}")
        if self.worker_status_enabled:
            hint_parts.append(f"worker_status_dir={self.worker_status_dir}")

        if isinstance(self._async_last_status, dict):
            try:
                last_resource = self._async_last_status.get("resource")
                last_workers = self._async_last_status.get("raw_pool_workers")
                top_workers = _top_worker_rss(last_workers, limit=4)
                hint_parts.append(
                    "last_status="
                    + json.dumps(
                        {
                            "collector_pid": self._async_last_status.get("collector_pid"),
                            "cache_size": self._async_last_status.get("cache_size"),
                            "jobs_total": self._async_last_status.get("jobs_total"),
                            "items_emitted": self._async_last_status.get("items_emitted"),
                            "sink": self._async_last_status.get("sink"),
                            "resource": last_resource,
                            "top_dataset_workers_by_rss": top_workers,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                pass
            fatal_error = self._async_last_status.get("fatal_error")
            if fatal_error:
                hint_parts.append(f"fatal_error={fatal_error}")
            fatal_traceback = self._async_last_status.get("fatal_traceback")
            if isinstance(fatal_traceback, str) and fatal_traceback:
                first_line = fatal_traceback.strip().splitlines()[-1]
                if first_line:
                    hint_parts.append(f"fatal_traceback_last_line={first_line}")
            fatal_report_path = self._async_last_status.get("fatal_report_path")
            if fatal_report_path:
                hint_parts.append(f"fatal_report_path={fatal_report_path}")

        message = f"Async collector process exited unexpectedly ({exit_reason})"
        if hint_parts:
            message = f"{message}. {' '.join(hint_parts)}"
        raise RuntimeError(message)

    def _build_crash_report_payload(
        self,
        *,
        exit_code: int | None,
        exit_reason: str,
        cgroup_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        process_pid = int(self._process.pid) if self._process is not None and self._process.pid is not None else None
        payload: dict[str, Any] = {
            "timestamp": float(time.time()),
            "exit_code": exit_code,
            "exit_reason": exit_reason,
            "collector_pid": process_pid,
            "collector_mode": self.collector_mode,
            "streaming_mode": self.streaming_mode,
            "collector_device": self.collector_device,
            "train_device": self.train_device,
            "jobs_total": int(self._jobs_total),
            "items_emitted": int(self._items_emitted),
            "jobs_by_model": dict(self._jobs_by_model),
            "cache_size": int(self.cache_size()),
            "async_events_dropped": int(self._async_events_dropped),
            "async_recent_jobs": list(self._async_recent_jobs),
            "async_last_status": self._async_last_status,
            "worker_status_enabled": bool(self.worker_status_enabled),
            "worker_status_dir": str(self.worker_status_dir),
            "host_snapshot": _collect_process_resource_snapshot(
                pid=os.getpid(),
                max_children=self.resource_snapshot_max_children,
            ),
            "cgroup": cgroup_snapshot or {},
            "cgroup_events_baseline": dict(self._cgroup_events_baseline),
        }
        payload["jobs_top_by_hwm_delta_mb"] = _top_jobs_by_memory_field(
            payload.get("async_recent_jobs"),
            field="memory_hwm_delta_mb",
            limit=8,
        )
        payload["jobs_top_by_rss_delta_mb"] = _top_jobs_by_memory_field(
            payload.get("async_recent_jobs"),
            field="memory_rss_delta_mb",
            limit=8,
        )
        if isinstance(self._async_last_status, dict):
            raw_pool_workers = self._async_last_status.get("raw_pool_workers")
            resource = self._async_last_status.get("resource")
            dataset_workers = _annotate_dataset_workers_with_resource(
                raw_pool_workers if isinstance(raw_pool_workers, dict) else None,
                resource if isinstance(resource, dict) else None,
            )
            if dataset_workers:
                payload["dataset_workers"] = dataset_workers
                payload["dataset_workers_top_by_rss"] = _top_worker_rss(dataset_workers, limit=8)
        if process_pid is not None:
            payload["collector_process_snapshot"] = _collect_process_resource_snapshot(
                pid=process_pid,
                max_children=self.resource_snapshot_max_children,
            )
        return payload

    def _maybe_write_crash_report(self, payload: dict[str, Any]) -> str | None:
        if not self.crash_report_enabled:
            return None
        path = self.crash_report_path
        _write_json_report(path, payload, logger=self.logger)
        return str(path)

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
            forced_terminate = False
            if self._stop_event is not None:
                self._stop_event.set()
            self._process.join(timeout=10)
            if self._process.is_alive():
                forced_terminate = True
                self._process.terminate()
                self._process.join(timeout=3)

            self._drain_status_queue()
            exit_code = self._process.exitcode
            if self.is_async_mode and exit_code not in (None, 0):
                exit_reason = _format_process_exit(exit_code)
                cgroup_snapshot = _read_cgroup_memory_snapshot() if self.include_cgroup_on_failure else {}
                crash_report_payload = self._build_crash_report_payload(
                    exit_code=exit_code,
                    exit_reason=exit_reason,
                    cgroup_snapshot=cgroup_snapshot,
                )
                crash_report_written = self._maybe_write_crash_report(crash_report_payload)
                if crash_report_written is not None:
                    self.logger.warning(
                        "Async collector exited with non-zero code during shutdown (%s, forced_terminate=%s). "
                        "crash_report_path=%s",
                        exit_reason,
                        forced_terminate,
                        crash_report_written,
                    )
                else:
                    self.logger.warning(
                        "Async collector exited with non-zero code during shutdown (%s, forced_terminate=%s).",
                        exit_reason,
                        forced_terminate,
                    )
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

        if self.parallel_inference_workers > 1:
            assert self._parallel_executor is not None
            in_flight: set[Future[CollectorJobStats]] = set()
            future_to_prepared: dict[Future[CollectorJobStats], _PreparedCollectorJob] = {}
            while not active_stop_event.is_set():
                self._ensure_prepared_job_producer_alive()
                submitted = False
                while (
                    len(in_flight) < self.parallel_inference_workers
                    and not active_stop_event.is_set()
                    and self._sink.needs_fill()
                ):
                    prepared = self._dequeue_prepared_job(timeout_s=0.2 if not in_flight else 0.05)
                    if prepared is None:
                        break
                    if self._is_model_failed(prepared.model_name):
                        self._discard_prepared_job(prepared)
                        continue
                    future = self._parallel_executor.submit(self._execute_prepared_parallel_job, prepared)
                    in_flight.add(future)
                    future_to_prepared[future] = prepared
                    submitted = True

                if not in_flight:
                    if not submitted:
                        time.sleep(0.05)
                    self._maybe_emit_status_event(event_type="heartbeat")
                    continue

                done, pending = wait(in_flight, timeout=0.1, return_when=FIRST_COMPLETED)
                in_flight = set(pending)
                for future in done:
                    prepared = future_to_prepared.pop(future, None)
                    try:
                        stats = future.result()
                    except Exception as exc:
                        self._mark_model_failed(prepared.model_name if prepared is not None else None, exc)
                        continue
                    self._record_completed_job(stats)

                self._maybe_emit_status_event(event_type="heartbeat")

            if in_flight:
                done, pending = wait(in_flight, timeout=30.0)
                for future in done:
                    prepared = future_to_prepared.pop(future, None)
                    try:
                        stats = future.result()
                    except Exception as exc:
                        self._mark_model_failed(prepared.model_name if prepared is not None else None, exc)
                        continue
                    self._record_completed_job(stats)
                for future in pending:
                    prepared = future_to_prepared.pop(future, None)
                    if future.cancel() and prepared is not None:
                        self._discard_prepared_job(prepared)
            return

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
        assert self._sink is not None

        job_resource_before: dict[str, Any] | None = None
        if self.emit_resource_snapshot:
            job_resource_before = _collect_process_resource_snapshot(
                pid=os.getpid(),
                max_children=self.resource_snapshot_max_children,
            )
        job_memory_before = _extract_job_memory_markers(job_resource_before)

        def _memory_payload_after() -> dict[str, float | None]:
            job_resource_after: dict[str, Any] | None = None
            if self.emit_resource_snapshot:
                job_resource_after = _collect_process_resource_snapshot(
                    pid=os.getpid(),
                    max_children=self.resource_snapshot_max_children,
                )
            job_memory_after = _extract_job_memory_markers(job_resource_after)
            return {
                "memory_rss_before_mb": job_memory_before.get("rss_mb"),
                "memory_rss_after_mb": job_memory_after.get("rss_mb"),
                "memory_rss_delta_mb": _delta_or_none(
                    job_memory_after.get("rss_mb"),
                    job_memory_before.get("rss_mb"),
                ),
                "memory_hwm_before_mb": job_memory_before.get("hwm_mb"),
                "memory_hwm_after_mb": job_memory_after.get("hwm_mb"),
                "memory_hwm_delta_mb": _delta_or_none(
                    job_memory_after.get("hwm_mb"),
                    job_memory_before.get("hwm_mb"),
                ),
                "memory_children_rss_before_mb": job_memory_before.get("children_rss_mb_sum"),
                "memory_children_rss_after_mb": job_memory_after.get("children_rss_mb_sum"),
                "memory_children_rss_delta_mb": _delta_or_none(
                    job_memory_after.get("children_rss_mb_sum"),
                    job_memory_before.get("children_rss_mb_sum"),
                ),
            }

        if not self._sink.needs_fill():
            return CollectorJobStats(
                model_name="none",
                num_images=0,
                num_layers=0,
                num_samples_emitted=0,
                dataset_mix={},
                duration_s=0.0,
                **_memory_payload_after(),
            )

        started = time.time()

        model_name = self._select_next_model_name()
        model_cfg = self.compat_index.get_model_cfg(model_name)
        batch_size = max(1, int(model_cfg.get("batch_size", 1)))

        raw_batch_fetch_started = time.time()
        image_refs, image_meta = self._raw_pool.sample_mixed_batch_refs(
            model_name=model_name,
            batch_size=batch_size,
            dataset_sampling_strategy=self.dataset_sampling_strategy,
            max_dataset_fraction_per_batch=self.max_dataset_fraction_per_batch,
        )
        raw_batch_fetch_s = time.time() - raw_batch_fetch_started

        if not image_refs:
            duration_s = time.time() - started
            return CollectorJobStats(
                model_name=model_name,
                num_images=0,
                num_layers=0,
                num_samples_emitted=0,
                dataset_mix={},
                duration_s=duration_s,
                raw_batch_fetch_s=raw_batch_fetch_s,
                **_memory_payload_after(),
            )

        pil_batch: list[Any] = []
        try:
            pil_batch, image_meta = self._raw_pool.materialize_image_refs(image_refs, image_meta)
            if not pil_batch:
                duration_s = time.time() - started
                return CollectorJobStats(
                    model_name=model_name,
                    num_images=0,
                    num_layers=0,
                    num_samples_emitted=0,
                    dataset_mix={},
                    duration_s=duration_s,
                    raw_batch_fetch_s=raw_batch_fetch_s,
                    **_memory_payload_after(),
                )

            infer_started = time.time()
            layer_records = self._get_execution_model_pool().run(model_name=model_name, pil_batch=pil_batch)
            model_infer_s = time.time() - infer_started

            atomize_started = time.time()
            run_id = self._next_run_id()
            if self.atomizer_process_enabled:
                self._enqueue_atomizer_task(
                    layer_records=layer_records,
                    image_meta_list=image_meta,
                    model_run_id=run_id,
                    model_name=model_name,
                )
                emitted = len(layer_records)
            else:
                emitted = 0
                for record in layer_records:
                    for shared_sample in atomize(
                        layer_record=record,
                        atom_cfg=self.atom_cfg,
                        image_meta_list=image_meta,
                        model_run_id=run_id,
                    ):
                        self._sink.emit(shared_sample)
                        emitted += 1
            atomize_emit_s = time.time() - atomize_started

            mix_counter = Counter(item.dataset_name for item in image_meta)

            duration_s = time.time() - started
            stats = CollectorJobStats(
                model_name=model_name,
                num_images=len(pil_batch),
                num_layers=len(layer_records),
                num_samples_emitted=emitted,
                dataset_mix=dict(mix_counter),
                duration_s=duration_s,
                raw_batch_fetch_s=raw_batch_fetch_s,
                model_infer_s=model_infer_s,
                atomize_emit_s=atomize_emit_s,
                **_memory_payload_after(),
            )
            return self._record_completed_job(stats)
        finally:
            self._raw_pool.release_image_refs(image_refs)

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
            "parallel_inference_workers": int(self.parallel_inference_workers),
            "parallel_model_pool_max_loaded_models": int(self.parallel_model_pool_max_loaded_models),
            "atomizer_process_enabled": bool(self.atomizer_process_enabled),
            "atomizer_process_alive": bool(self._atomizer_process is not None and self._atomizer_process.is_alive()),
            "cache_size": self.cache_size(),
            "jobs_total": self._jobs_total,
            "jobs_by_model": dict(self._jobs_by_model),
            "items_emitted": self._items_emitted,
            "scheduler": self._scheduler.stats() if self._scheduler else None,
            "sink": self._sink.stats() if self._sink else None,
            "status_emit_interval_s": self.status_emit_interval_s,
            "status_queue_max_items": self.status_queue_max_items,
            "worker_status_enabled": bool(self.worker_status_enabled),
            "worker_status_dir": str(self.worker_status_dir),
        }

        if self.is_async_mode:
            payload["async_process_alive"] = bool(self._process is not None and self._process.is_alive())
            payload["async_events_dropped"] = int(self._async_events_dropped)
            payload["async_recent_jobs"] = list(self._async_recent_jobs)
            payload["async_last_status"] = self._async_last_status
            payload["async_last_atomizer_event"] = self._async_last_atomizer_event

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
        if self.worker_status_enabled:
            self.logger.info("Dataset worker status dir: %s", self.worker_status_dir)

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
            self._runtime_device = runtime_device
            self._runtime_local_only = bool(collector_cfg.get("runtime_local_only", False))
            self._runtime_release_device_on_unload = release_device_on_unload
            self._runtime_empty_cuda_cache_on_unload = empty_cuda_cache_on_unload

            if self.parallel_inference_workers <= 1:
                self._model_pool = ModelPool(
                    global_cfg=self.cfg,
                    model_cfgs=model_cfgs,
                    device_override=runtime_device,
                    max_loaded_models=max_loaded,
                    runtime_local_only=self._runtime_local_only,
                    release_device_on_unload=release_device_on_unload,
                    empty_cuda_cache_on_unload=empty_cuda_cache_on_unload,
                    load_report=self._load_report,
                )
            else:
                self._parallel_executor = ThreadPoolExecutor(
                    max_workers=self.parallel_inference_workers,
                    thread_name_prefix="collector_infer",
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
            if self.atomizer_process_enabled:
                self._start_atomizer_process()
            if self.parallel_inference_workers > 1:
                self._start_prepared_job_producer()
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

    def _start_prepared_job_producer(self) -> None:
        if self.parallel_inference_workers <= 1:
            return
        if self._sink is None:
            raise RuntimeError("collector sink must be initialized before starting prepared-job producer")
        if self._prepared_job_queue is None:
            self._prepared_job_queue = ThreadQueue(maxsize=self.prepared_job_queue_max_items)
        if self._prepared_job_stop_event is None:
            self._prepared_job_stop_event = threading.Event()
        self._prepared_job_stop_event.clear()
        self._prepared_job_producer_error = None
        if self._prepared_job_producer_thread is not None and self._prepared_job_producer_thread.is_alive():
            return
        self._prepared_job_producer_thread = threading.Thread(
            target=self._prepared_job_producer_main,
            name="collector_prepare",
            daemon=True,
        )
        self._prepared_job_producer_thread.start()

    def _prepared_job_producer_main(self) -> None:
        assert self._prepared_job_queue is not None
        assert self._prepared_job_stop_event is not None
        assert self._sink is not None
        prepared_job_queue = self._prepared_job_queue
        stop_event = self._prepared_job_stop_event
        sink = self._sink

        try:
            while not stop_event.is_set():
                if not sink.needs_fill():
                    time.sleep(0.02)
                    continue
                if prepared_job_queue.full():
                    time.sleep(0.02)
                    continue

                prepared = self._prepare_parallel_job()
                if prepared is None:
                    time.sleep(0.02)
                    continue
                if self._is_model_failed(prepared.model_name):
                    self._discard_prepared_job(prepared)
                    continue

                while not stop_event.is_set():
                    try:
                        prepared_job_queue.put(prepared, timeout=0.1)
                        break
                    except Full:
                        continue
        except BaseException as exc:
            self._prepared_job_producer_error = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            self.logger.exception("Prepared-job producer thread failed")

    def _ensure_prepared_job_producer_alive(self) -> None:
        if self.parallel_inference_workers <= 1:
            return
        if self._prepared_job_producer_error is not None:
            raise RuntimeError(
                "collector prepared-job producer thread failed: "
                f"{self._prepared_job_producer_error.get('error')}"
            )
        if self._prepared_job_producer_thread is None or not self._prepared_job_producer_thread.is_alive():
            raise RuntimeError("collector prepared-job producer thread exited unexpectedly")

    def _dequeue_prepared_job(self, timeout_s: float) -> _PreparedCollectorJob | None:
        if self._prepared_job_queue is None:
            return None
        try:
            return self._prepared_job_queue.get(timeout=max(0.0, float(timeout_s)))
        except Empty:
            return None

    def _discard_prepared_job(self, prepared: _PreparedCollectorJob) -> None:
        if self._raw_pool is None:
            return
        try:
            self._raw_pool.release_image_refs(prepared.image_refs)
        except Exception as exc:
            self.logger.warning(
                "Failed to release discarded prepared job refs for model '%s': %s",
                prepared.model_name,
                exc,
            )

    def _is_model_failed(self, model_name: str) -> bool:
        with self._stats_lock:
            return str(model_name) in self._failed_models

    def _shutdown_runtime_components(self) -> None:
        if self._prepared_job_stop_event is not None:
            self._prepared_job_stop_event.set()
        if self._prepared_job_producer_thread is not None:
            self._prepared_job_producer_thread.join(timeout=5)
            if self._prepared_job_producer_thread.is_alive():
                self.logger.warning("Prepared-job producer thread did not stop before collector shutdown")
            self._prepared_job_producer_thread = None
        self._prepared_job_stop_event = None
        self._prepared_job_producer_error = None
        if self._prepared_job_queue is not None:
            while True:
                try:
                    prepared = self._prepared_job_queue.get_nowait()
                except Empty:
                    break
                except Exception:
                    break
                if isinstance(prepared, _PreparedCollectorJob):
                    self._discard_prepared_job(prepared)
            self._prepared_job_queue = None

        if self._atomizer_stop_event is not None:
            self._atomizer_stop_event.set()
        if self._atomizer_process is not None:
            self._atomizer_process.join(timeout=10)
            if self._atomizer_process.is_alive():
                self._atomizer_process.terminate()
                self._atomizer_process.join(timeout=3)
            self._atomizer_process = None
        if self._atomizer_task_queue is not None:
            try:
                self._atomizer_task_queue.close()
            except Exception:
                pass
            self._atomizer_task_queue = None
        self._atomizer_stop_event = None

        if self._sink is not None:
            try:
                self._sink.close()
            except Exception as exc:
                self.logger.warning("Failed to close collector sink: %s", exc)
            self._sink = None

        if self._parallel_executor is not None:
            self._parallel_executor.shutdown(wait=True, cancel_futures=False)
            self._parallel_executor = None

        if self._model_pool is not None:
            self._model_pool.unload_all()
            self._model_pool = None

        with self._parallel_model_pools_lock:
            for pool in self._parallel_model_pools.values():
                try:
                    pool.unload_all()
                except Exception as exc:
                    self.logger.warning("Failed to unload parallel collector model pool: %s", exc)
            self._parallel_model_pools.clear()

        if self._raw_pool is not None:
            self._raw_pool.shutdown()
            self._raw_pool = None

        self._scheduler = None
        self._runtime_device = None
        self._runtime_ready = False

    def _start_atomizer_process(self) -> None:
        if self.streaming_mode != "none":
            raise RuntimeError("atomizer process requires streaming.mode=none")
        if self.cache is None:
            raise RuntimeError("atomizer process requires in-memory SharedSampleCache")
        if self._atomizer_process is not None and self._atomizer_process.is_alive():
            return

        self._atomizer_task_queue = self._ctx.Queue(maxsize=self.atomizer_queue_max_items)
        self._atomizer_stop_event = self._ctx.Event()
        self._atomizer_process = self._ctx.Process(
            target=collector_atomizer_process_main,
            args=(
                self.cfg_dict,
                self.cache,
                self.atom_cfg,
                self._atomizer_task_queue,
                self._atomizer_stop_event,
                self._status_queue,
            ),
            daemon=False,
            name="collector_atomizer",
        )
        self._atomizer_process.start()
        self.logger.info("Started collector atomizer process pid=%s", self._atomizer_process.pid)

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
        with self._stats_lock:
            self._model_run_id += 1
            return self._model_run_id

    def _get_execution_model_pool(self) -> ModelPool:
        if self.parallel_inference_workers <= 1:
            if self._model_pool is None:
                raise RuntimeError("collector model pool is not initialized")
            return self._model_pool

        thread_id = threading.get_ident()
        with self._parallel_model_pools_lock:
            pool = self._parallel_model_pools.get(thread_id)
            if pool is not None:
                return pool
            pool = ModelPool(
                global_cfg=self.cfg,
                model_cfgs=self.compat_index.get_model_cfgs(),
                device_override=self._runtime_device,
                max_loaded_models=self.parallel_model_pool_max_loaded_models,
                runtime_local_only=self._runtime_local_only,
                release_device_on_unload=self._runtime_release_device_on_unload,
                empty_cuda_cache_on_unload=self._runtime_empty_cuda_cache_on_unload,
                load_report=self._load_report,
            )
            self._parallel_model_pools[thread_id] = pool
            return pool

    def _ensure_atomizer_alive(self) -> None:
        if not self.atomizer_process_enabled:
            return
        if self._atomizer_process is None:
            raise RuntimeError("collector atomizer process is not initialized")
        if self._atomizer_process.is_alive():
            return
        raise RuntimeError(f"collector atomizer process exited unexpectedly ({_format_process_exit(self._atomizer_process.exitcode)})")

    def _enqueue_atomizer_task(
        self,
        *,
        layer_records: list[Any],
        image_meta_list: list[Any],
        model_run_id: int,
        model_name: str,
    ) -> None:
        self._ensure_atomizer_alive()
        if self._atomizer_task_queue is None:
            raise RuntimeError("collector atomizer task queue is not initialized")

        layer_record_refs = share_layer_record_refs(layer_records)
        payload = {
            "layer_record_refs": layer_record_refs,
            "image_meta_list": image_meta_list,
            "model_run_id": int(model_run_id),
            "model_name": str(model_name),
            "timestamp": float(time.time()),
        }
        enqueued = False
        try:
            while True:
                try:
                    self._atomizer_task_queue.put(payload, timeout=0.25)
                    enqueued = True
                    return
                except Full:
                    self._ensure_atomizer_alive()
                    continue
        finally:
            if not enqueued:
                cleanup_shared_layer_record_refs(layer_record_refs)

    def _prepare_parallel_job(self) -> _PreparedCollectorJob | None:
        assert self._scheduler is not None
        assert self._raw_pool is not None

        started = time.time()
        model_name = self._select_next_model_name()
        model_cfg = self.compat_index.get_model_cfg(model_name)
        batch_size = max(1, int(model_cfg.get("batch_size", 1)))

        image_refs, image_meta = self._raw_pool.sample_mixed_batch_refs(
            model_name=model_name,
            batch_size=batch_size,
            dataset_sampling_strategy=self.dataset_sampling_strategy,
            max_dataset_fraction_per_batch=self.max_dataset_fraction_per_batch,
        )
        raw_batch_fetch_s = time.time() - started
        if not image_refs:
            return None

        dataset_mix = dict(Counter(item.dataset_name for item in image_meta))
        return _PreparedCollectorJob(
            model_name=model_name,
            image_refs=image_refs,
            image_meta=image_meta,
            dataset_mix=dataset_mix,
            started_at=started,
            raw_batch_fetch_s=raw_batch_fetch_s,
        )

    def _execute_prepared_parallel_job(self, prepared: _PreparedCollectorJob) -> CollectorJobStats:
        assert self._sink is not None

        model_pool = self._get_execution_model_pool()
        pil_batch: list[Any] = []
        try:
            pil_batch, image_meta = self._raw_pool.materialize_image_refs(prepared.image_refs, prepared.image_meta)
            if not pil_batch:
                return CollectorJobStats(
                    model_name=prepared.model_name,
                    num_images=0,
                    num_layers=0,
                    num_samples_emitted=0,
                    dataset_mix=prepared.dataset_mix,
                    duration_s=time.time() - prepared.started_at,
                    raw_batch_fetch_s=prepared.raw_batch_fetch_s,
                )

            infer_started = time.time()
            layer_records = model_pool.run(model_name=prepared.model_name, pil_batch=pil_batch)
            model_infer_s = time.time() - infer_started

            atomize_started = time.time()
            run_id = self._next_run_id()
            if self.atomizer_process_enabled:
                self._enqueue_atomizer_task(
                    layer_records=layer_records,
                    image_meta_list=image_meta,
                    model_run_id=run_id,
                    model_name=prepared.model_name,
                )
                emitted = len(layer_records)
            else:
                emitted = 0
                for record in layer_records:
                    for shared_sample in atomize(
                        layer_record=record,
                        atom_cfg=self.atom_cfg,
                        image_meta_list=image_meta,
                        model_run_id=run_id,
                    ):
                        self._sink.emit(shared_sample)
                        emitted += 1
            atomize_emit_s = time.time() - atomize_started

            return CollectorJobStats(
                model_name=prepared.model_name,
                num_images=len(pil_batch),
                num_layers=len(layer_records),
                num_samples_emitted=emitted,
                dataset_mix=prepared.dataset_mix,
                duration_s=time.time() - prepared.started_at,
                raw_batch_fetch_s=prepared.raw_batch_fetch_s,
                model_infer_s=model_infer_s,
                atomize_emit_s=atomize_emit_s,
            )
        finally:
            self._raw_pool.release_image_refs(prepared.image_refs)

    def _record_completed_job(self, stats: CollectorJobStats) -> CollectorJobStats:
        with self._stats_lock:
            self._jobs_total += 1
            self._jobs_by_model[stats.model_name] += 1
            self._items_emitted += int(stats.num_samples_emitted)

        self.logger.info(
            "collector job model=%s images=%s layers=%s emitted=%s size=%s mode=%s "
            "raw_batch_fetch_s=%.3f model_infer_s=%.3f atomize_emit_s=%.3f total_s=%.3f "
            "rss_after_mb=%s rss_delta_mb=%s hwm_delta_mb=%s child_rss_delta_mb=%s",
            stats.model_name,
            stats.num_images,
            stats.num_layers,
            stats.num_samples_emitted,
            self.cache_size(),
            self.streaming_mode,
            float(stats.raw_batch_fetch_s),
            float(stats.model_infer_s),
            float(stats.atomize_emit_s),
            float(stats.duration_s),
            _fmt_mb(stats.memory_rss_after_mb),
            _fmt_mb(stats.memory_rss_delta_mb),
            _fmt_mb(stats.memory_hwm_delta_mb),
            _fmt_mb(stats.memory_children_rss_delta_mb),
        )
        self._emit_status_event(event_type="job", job_stats=stats)
        return stats

    def _select_next_model_name(self) -> str:
        assert self._scheduler is not None
        all_models = self.compat_index.get_models()
        if not all_models:
            raise RuntimeError("collector has no compatible models to schedule")
        with self._stats_lock:
            failed_models = set(self._failed_models)
        if len(failed_models) >= len(all_models):
            raise RuntimeError(
                "collector exhausted all compatible models after runtime failures: "
                f"{sorted(failed_models)}"
            )
        for _ in range(len(all_models) * 2):
            model_name = self._scheduler.next_model()
            if model_name not in failed_models:
                return model_name
        for model_name in all_models:
            if model_name not in failed_models:
                return model_name
        raise RuntimeError(
            "collector failed to select an active model after runtime failures: "
            f"{sorted(failed_models)}"
        )

    def _mark_model_failed(self, model_name: str | None, exc: BaseException) -> None:
        if model_name is None:
            self.logger.exception("Collector job failed for unknown model", exc_info=exc)
            return
        with self._stats_lock:
            first_failure = model_name not in self._failed_models
            self._failed_models.add(model_name)
        self._purge_failed_model_from_pools(model_name)
        if first_failure:
            self.logger.exception(
                "Disabling collector model after runtime failure: model=%s error=%s",
                model_name,
                exc,
                exc_info=exc,
            )
        else:
            self.logger.warning("Collector model remains disabled after repeated failure: model=%s error=%s", model_name, exc)
        self._emit_status_event(event_type="heartbeat", job_stats=None)

    def _purge_failed_model_from_pools(self, model_name: str) -> None:
        if self._model_pool is not None:
            self._model_pool.unload_model(model_name)
        with self._parallel_model_pools_lock:
            for pool in self._parallel_model_pools.values():
                pool.unload_model(model_name)

    def _maybe_emit_status_event(self, event_type: str) -> None:
        now = time.time()
        if now - self._last_status_emit_ts < self.status_emit_interval_s:
            return
        self._emit_status_event(event_type=event_type, job_stats=None)

    def _emit_status_event(self, event_type: str, job_stats: CollectorJobStats | None) -> None:
        with self._stats_lock:
            jobs_total = int(self._jobs_total)
            jobs_by_model = {name: int(value) for name, value in self._jobs_by_model.items()}
            items_emitted = int(self._items_emitted)
            failed_models = sorted(self._failed_models)
        payload: dict[str, Any] = {
            "type": str(event_type),
            "timestamp": float(time.time()),
            "collector_pid": int(os.getpid()),
            "mode": self.collector_mode,
            "streaming_mode": self.streaming_mode,
            "parallel_inference_workers": int(self.parallel_inference_workers),
            "prepared_job_queue_size": int(0 if self._prepared_job_queue is None else self._prepared_job_queue.qsize()),
            "prepared_job_producer_alive": bool(
                self._prepared_job_producer_thread is not None and self._prepared_job_producer_thread.is_alive()
            ),
            "atomizer_process_enabled": bool(self.atomizer_process_enabled),
            "atomizer_process_alive": bool(self._atomizer_process is not None and self._atomizer_process.is_alive()),
            "cache_size": int(self.cache_size()),
            "jobs_total": jobs_total,
            "jobs_by_model": jobs_by_model,
            "items_emitted": items_emitted,
            "failed_models": failed_models,
            "scheduler": self._scheduler.stats() if self._scheduler else None,
            "sink": self._sink.stats() if self._sink else None,
            "events_dropped": int(self._async_events_dropped),
        }
        if self._prepared_job_producer_error is not None:
            payload["prepared_job_producer_error"] = dict(self._prepared_job_producer_error)
        if self.emit_resource_snapshot:
            payload["resource"] = _collect_process_resource_snapshot(
                pid=os.getpid(),
                max_children=self.resource_snapshot_max_children,
            )
        if self._raw_pool is not None:
            try:
                raw_pool_workers = self._raw_pool.worker_processes()
                payload["raw_pool_workers"] = _annotate_dataset_workers_with_resource(
                    raw_pool_workers,
                    payload.get("resource") if isinstance(payload.get("resource"), dict) else None,
                )
            except Exception as exc:
                payload["raw_pool_workers_error"] = str(exc)
        if job_stats is not None:
            payload["job_stats"] = asdict(job_stats)

        self._write_dataset_worker_status_files(payload)

        if self._status_queue is None:
            self._last_status_emit_ts = payload["timestamp"]
            return

        try:
            self._status_queue.put_nowait(payload)
            self._last_status_emit_ts = payload["timestamp"]
        except Full:
            self._async_events_dropped += 1
        except Exception:
            pass

    def _write_dataset_worker_status_files(self, payload: dict[str, Any]) -> None:
        if not self.worker_status_enabled:
            return
        workers = payload.get("raw_pool_workers")
        if not isinstance(workers, dict):
            return

        timestamp = float(payload.get("timestamp", time.time()))
        collector_pid = _safe_int(payload.get("collector_pid"))
        event_type = str(payload.get("type", "status"))
        cache_size = _safe_int(payload.get("cache_size"))
        jobs_total = _safe_int(payload.get("jobs_total"))
        items_emitted = _safe_int(payload.get("items_emitted"))

        all_rows_path = self.worker_status_dir / "_all_workers.jsonl"
        for dataset_name, info in workers.items():
            if not isinstance(info, dict):
                continue
            row = {
                "timestamp": timestamp,
                "event_type": event_type,
                "dataset": str(dataset_name),
                "collector_pid": collector_pid,
                "cache_size": cache_size,
                "jobs_total": jobs_total,
                "items_emitted": items_emitted,
                **info,
            }
            safe_dataset_name = str(dataset_name).replace("/", "__")
            dataset_path = self.worker_status_dir / f"{safe_dataset_name}.jsonl"
            _append_jsonl_record(dataset_path, row, logger=self.logger)
            _append_jsonl_record(all_rows_path, row, logger=self.logger)

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

            payload_type = str(payload.get("type", ""))
            if payload_type.startswith("atomizer_"):
                self._async_last_atomizer_event = payload
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
                    "raw_batch_fetch_s": float(job_stats.get("raw_batch_fetch_s", 0.0)),
                    "model_infer_s": float(job_stats.get("model_infer_s", 0.0)),
                    "atomize_emit_s": float(job_stats.get("atomize_emit_s", 0.0)),
                    "memory_rss_before_mb": _safe_float(job_stats.get("memory_rss_before_mb")),
                    "memory_rss_after_mb": _safe_float(job_stats.get("memory_rss_after_mb")),
                    "memory_rss_delta_mb": _safe_float(job_stats.get("memory_rss_delta_mb")),
                    "memory_hwm_before_mb": _safe_float(job_stats.get("memory_hwm_before_mb")),
                    "memory_hwm_after_mb": _safe_float(job_stats.get("memory_hwm_after_mb")),
                    "memory_hwm_delta_mb": _safe_float(job_stats.get("memory_hwm_delta_mb")),
                    "memory_children_rss_before_mb": _safe_float(job_stats.get("memory_children_rss_before_mb")),
                    "memory_children_rss_after_mb": _safe_float(job_stats.get("memory_children_rss_after_mb")),
                    "memory_children_rss_delta_mb": _safe_float(job_stats.get("memory_children_rss_delta_mb")),
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
    forensics_cfg_dict = dict(cfg_dict)
    if "train" not in forensics_cfg_dict and isinstance(forensics_cfg_dict.get("mini_train"), dict):
        forensics_cfg_dict["train"] = dict(forensics_cfg_dict["mini_train"])

    try:
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass

        maybe_redirect_stdio(forensics_cfg_dict, role="collector_process", section="train")
        log_path = configure_process_logging(cfg=cfg_dict, role="collector_process", force=True)
        logger = logging.getLogger("collector_process")
        maybe_enable_core_dumps(forensics_cfg_dict, section="train", logger=logger)
        logger.info("Run log file: %s", log_path)
        fault_path = _enable_fault_handler_log(log_path, label="collector_process", logger=logger)
        if fault_path is not None:
            logger.info("Collector fault log file: %s", fault_path)

        service = CollectorService(cfg=cfg_dict, cache=cache, status_queue=status_queue)
        try:
            service.run_forever(stop_event=stop_event)
        except KeyboardInterrupt:
            logger.info("Collector process interrupted")
        finally:
            service.shutdown()
    except BaseException as exc:
        logger = logging.getLogger("collector_process")
        traceback_text = traceback.format_exc()
        report_path = emit_fatal_report(
            forensics_cfg_dict,
            role="collector_process",
            error=str(exc),
            traceback_text=traceback_text,
            extra={
                "collector_mode": str(cfg_dict.get("collector", {}).get("mode", "unknown")),
            },
            section="train",
        )
        logger.exception("Collector process failed with unhandled exception")
        if status_queue is not None:
            try:
                status_queue.put_nowait(
                    {
                        "type": "fatal_exception",
                        "timestamp": float(time.time()),
                        "collector_pid": int(os.getpid()),
                        "fatal_error": str(exc),
                        "fatal_traceback": traceback_text,
                        "fatal_report_path": report_path,
                    }
                )
            except Exception:
                pass
        raise


def collector_atomizer_process_main(
    cfg_dict: dict[str, Any],
    cache: SharedSampleCache,
    atom_cfg: dict[str, Any],
    task_queue: Any,
    stop_event: Any,
    status_queue: Any | None = None,
) -> None:
    log_path = configure_process_logging(cfg=cfg_dict, role="collector_atomizer", force=True)
    logger = logging.getLogger("collector_atomizer")
    logger.info("Run log file: %s", log_path)
    try:
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass

        while True:
            try:
                if stop_event.is_set():
                    payload = task_queue.get_nowait()
                else:
                    payload = task_queue.get(timeout=0.2)
            except Empty:
                if stop_event.is_set():
                    break
                continue

            if not isinstance(payload, dict):
                continue

            layer_records = payload.get("layer_records")
            layer_record_refs = payload.get("layer_record_refs")
            image_meta_list = payload.get("image_meta_list")
            model_run_id = int(payload.get("model_run_id", 0))
            model_name = str(payload.get("model_name", ""))

            if isinstance(layer_record_refs, list):
                layer_records = restore_layer_record_refs(layer_record_refs, release=True)
            elif not isinstance(layer_records, list):
                continue
            if not isinstance(image_meta_list, list):
                image_meta_list = []

            emitted = 0
            started_at = time.time()
            for record in layer_records:
                for shared_sample in atomize(
                    layer_record=record,
                    atom_cfg=atom_cfg,
                    image_meta_list=image_meta_list,
                    model_run_id=model_run_id,
                ):
                    cache.put(shared_sample)
                    emitted += 1

            if status_queue is not None:
                try:
                    status_queue.put_nowait(
                        {
                            "type": "atomizer_job",
                            "timestamp": float(time.time()),
                            "model_name": model_name,
                            "model_run_id": model_run_id,
                            "num_layers": len(layer_records),
                            "num_samples_emitted": emitted,
                            "duration_s": float(time.time() - started_at),
                            "atomizer_pid": int(os.getpid()),
                        }
                    )
                except Exception:
                    pass
    except BaseException as exc:
        logger.exception("Collector atomizer process failed with unhandled exception")
        if status_queue is not None:
            try:
                status_queue.put_nowait(
                    {
                        "type": "atomizer_fatal_exception",
                        "timestamp": float(time.time()),
                        "atomizer_pid": int(os.getpid()),
                        "fatal_error": str(exc),
                        "fatal_traceback": traceback.format_exc(),
                    }
                )
            except Exception:
                pass
        raise


def stats_to_dict_list(items: list[CollectorJobStats]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
